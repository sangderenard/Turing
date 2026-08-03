"""Target-neutral compiled-shell control structure and source rendering.

The planner owns control flow.  Backends render this structure; they must not
rediscover loops, reorder scheduled regions, or substitute a host-language
coordinator after planning has selected a compiled target.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping

from .deployment_frame import DeploymentFrame, DeploymentJoin


class ControlTarget(str, Enum):
    PYTHON = "python"
    C = "c"
    GLSL = "glsl"
    # Fortran renders the same control structure with `do`/`select case`.  It
    # is worth having as a launch environment because its arrays cannot alias,
    # which is the freedom the C and LLVM routes must assert explicitly.
    FORTRAN = "fortran"


@dataclass(frozen=True)
class StatementBlock:
    """Already-lowered statements in planner-approved execution order."""

    lines: tuple[str, ...]


@dataclass(frozen=True)
class SequenceBlock:
    blocks: tuple["ControlBlock", ...]


@dataclass(frozen=True)
class ControlExpression:
    """Typed scalar expression evaluated by the compiled control backend."""

    op: str
    operands: tuple["ControlExpression", ...] = ()
    value_id: int | None = None
    literal: bool | int | float | None = None


@dataclass(frozen=True)
class LoopBlock:
    induction: str
    start: str
    stop: str
    step: str
    body: "ControlBlock"
    # Storage aliases remain listed on ControlProgram for allocation, while
    # the actual carried-state commits belong only to this lexical loop.
    carried_aliases: tuple[tuple[int, int], ...] = ()
    # The planner has proved that iterations communicate only through
    # induction-indexed publications.  A parallel backend may map one
    # iteration to one workgroup; ordinary renderers retain a serial loop.
    parallel_iterations: bool = False
    # A C dispatch shell dissolves the loop into one-iteration shader calls.
    dispatch_shell: str = "glsl"
    # Stable entry in ControlProgram.recursion_regions.  The CFG/SSA lowerer
    # uses this to associate the loop header, Phi nodes, and latch backedge
    # with the ProcessGraph SCC from which this loop was retained.
    recursion_region_id: int | None = None
    schedule_preference: str = "alap"

    def __post_init__(self) -> None:
        preference = str(self.schedule_preference).lower()
        if preference not in {"asap", "alap"}:
            raise ValueError(
                "loop schedule preference must be 'asap' or 'alap'"
            )
        object.__setattr__(self, "schedule_preference", preference)


@dataclass(frozen=True)
class WhileBlock:
    """Condition-controlled loop with an explicitly scheduled predicate.

    ``condition`` is run before the first test and again at the latch.  This
    lets a numerical region compute the predicate without smuggling Python
    evaluation into a compiled shell.  The predicate itself remains an
    ordinary resident value shared by every backend.
    """

    predicate_value_id: int
    condition: "ControlBlock"
    body: "ControlBlock"
    carried_aliases: tuple[tuple[int, int], ...] = ()
    recursion_region_id: int | None = None
    predicate_expression: ControlExpression | None = None


@dataclass(frozen=True)
class LoopControlBlock:
    """A planner-owned ``break`` or ``continue`` edge.

    When ``predicate_value_id`` is absent the edge is unconditional.  A
    conditional edge branches only when the resident predicate is true.
    """

    action: str
    predicate_value_id: int | None = None
    expect_true: bool = True
    predicate_expression: ControlExpression | None = None

    def __post_init__(self) -> None:
        if self.action not in {"break", "continue"}:
            raise ValueError("loop control action must be break or continue")


@dataclass(frozen=True)
class StateMachineTick:
    """One compiled state transition, not a host polling loop."""

    state: str
    cases: tuple[tuple[str, "ControlBlock"], ...]
    default: "ControlBlock | None" = None


@dataclass(frozen=True)
class ParallelDeployment:
    """Independent scheduled lanes available to one backend deployment."""

    lanes: tuple["ControlBlock", ...]
    schedule_preference: str = "alap"

    def __post_init__(self) -> None:
        preference = str(self.schedule_preference).lower()
        if preference not in {"asap", "alap"}:
            raise ValueError(
                "parallel schedule preference must be 'asap' or 'alap'"
            )
        object.__setattr__(self, "schedule_preference", preference)


@dataclass(frozen=True)
class ControlDeploymentLane:
    """One proven-independent lane in a backend-neutral deployment region."""

    index: int
    region_indices: tuple[int, ...] = ()
    value_ids: tuple[int, ...] = ()
    source_node_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ControlDeploymentRegion:
    """Durable parallelism evidence beside the lexical Control IR.

    The record grants a later deployment pass permission to schedule the
    listed lanes concurrently.  It deliberately does not select GLSL, SIMD,
    threads, or any other backend, and the lexical control tree remains the
    serial fallback.
    """

    region_id: int
    kind: str
    schedule: str
    schedule_preference: str = "alap"
    lanes: tuple[ControlDeploymentLane, ...] = ()
    iteration_space: tuple[str, str, str] | None = None
    carried_aliases: tuple[tuple[int, int], ...] = ()
    recursion_region_id: int | None = None
    origin: str = "control_ir"
    source_loop_node_id: int | None = None
    scale: int = 1
    join: DeploymentJoin = DeploymentJoin()

    def __post_init__(self) -> None:
        preference = str(self.schedule_preference).lower()
        if preference not in {"asap", "alap"}:
            raise ValueError(
                "deployment schedule preference must be 'asap' or 'alap'"
            )
        object.__setattr__(self, "schedule_preference", preference)
        indices = tuple(int(lane.index) for lane in self.lanes)
        if indices != tuple(range(len(self.lanes))):
            raise ValueError(
                "deployment lane indices must be contiguous from zero"
            )
        DeploymentFrame(self.region_id, self.scale, self.join)

    @property
    def frame(self) -> DeploymentFrame:
        return DeploymentFrame(self.region_id, self.scale, self.join)


@dataclass(frozen=True)
class CallBlock:
    """Planner-owned nested closure invocation with explicit value bindings."""

    callsite_id: int
    callee: "ControlBlock"
    argument_bindings: tuple[tuple[int, int], ...] = ()
    result_bindings: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ValidationBlock:
    """Device-side predicate whose failure is reported through shell errors."""

    predicate_value_id: int
    error_code: int
    expect_true: bool = True


@dataclass(frozen=True)
class StreamPublishBlock:
    """Publish one value range to a planner-owned backpressured stream.

    This is a logical output operation.  It does not imply Python ``yield``,
    host materialization, or a backend callback.  A backend must reserve
    resident output capacity, copy the payload, publish its descriptor, and
    leave the compiled state machine suspended when capacity is exhausted.
    """

    stream_id: int
    value_id: int
    count_value_id: int | None = None
    predicate_value_id: int | None = None
    final: bool = False


ControlBlock = (
    StatementBlock
    | SequenceBlock
    | LoopBlock
    | WhileBlock
    | LoopControlBlock
    | StateMachineTick
    | ParallelDeployment
    | CallBlock
    | ValidationBlock
    | StreamPublishBlock
)


@dataclass(frozen=True)
class RecursionRegion:
    """One cached strongly connected ProcessGraph region."""

    region_id: int
    kind: str
    lower_as: str
    members: tuple[int, ...]
    control_ir: bool = True
    control_members: tuple[int, ...] = ()
    incoming: tuple[tuple[int, int, str], ...] = ()
    outgoing: tuple[tuple[int, int, str], ...] = ()
    feedback: tuple[tuple[int, int, str], ...] = ()


@dataclass(frozen=True)
class ControlProgram:
    root: ControlBlock
    region_indices: tuple[int, ...] = ()
    uniforms: tuple["ControlUniform", ...] = ()
    value_aliases: tuple[tuple[int, int], ...] = ()
    # (iterable value id, loop-target value id, induction variable)
    # The backend uses this to bind an iterable element inside compiled
    # control flow.  It is not a request for the host to assign a Python name.
    iterable_bindings: tuple[tuple[int, int, str], ...] = ()
    # (iterable value id, target value id, induction, source-defined values)
    static_iterable_bindings: tuple[
        tuple[int, int, str, tuple[object, ...]], ...
    ] = ()
    # (per-iteration source value, resident collection value, induction,
    # source start). Backends write the source into an indexed resident range;
    # they never reconstruct a Python list from observed iteration values.
    collection_bindings: tuple[tuple[int, int, str, int], ...] = ()
    # (aggregate value id, target value id, induction, resident source ids).
    # This is a planner identity list, not a reconstructed Python container.
    closure_iterable_bindings: tuple[
        tuple[int, int, str, tuple[int, ...]], ...
    ] = ()
    recursion_regions: tuple[RecursionRegion, ...] = ()
    # Scheduling permissions survive independently of source syntax.  This
    # is where an evaporated/unrolled loop can condense instead of becoming
    # indistinguishable straight-line work.
    deployment_regions: tuple[ControlDeploymentRegion, ...] = ()


@dataclass(frozen=True)
class ControlUniform:
    name: str
    value_id: int
    dtype: str = "int"


@dataclass(frozen=True)
class RegionCode:
    """One backend implementation selected for a logical region.

    ``body`` is source in the interior's own language.  ``launch_body`` is
    optional C source which invokes an already-compiled GLSL interior through
    a passed OpenGL context.  Keeping those two representations separate is
    critical: GLSL source must never be pasted into the C function and
    mistaken for C control code.  This is a specific C-shell capability, not
    general permission for every non-GLSL language to launch arbitrary
    interiors.  A GLSL shell cannot contain or launch C and cannot drop out to
    a host launcher.
    """

    region_index: int
    target: ControlTarget
    body: StatementBlock
    launch_body: StatementBlock | None = None


def _indent(lines: Iterable[str], spaces: int = 4) -> tuple[str, ...]:
    prefix = " " * spaces
    return tuple(prefix + line if line else line for line in lines)


def _render_loop(block: LoopBlock, target: ControlTarget) -> tuple[str, ...]:
    body = _indent(render_control_block(block.body, target))
    if target is ControlTarget.GLSL and block.dispatch_shell == "c":
        return (
            f"int {block.induction} = u_dispatch_iteration;",
            *body,
        )
    if target is ControlTarget.GLSL and block.parallel_iterations:
        return (
            f"int {block.induction} = int({block.start}) + "
            f"int(gl_WorkGroupID.x) * int({block.step});",
            f"if ({block.induction} < int({block.stop})) {{",
            *body,
            "}",
        )
    if target is ControlTarget.PYTHON:
        return (
            f"for {block.induction} in range("
            f"{block.start}, {block.stop}, {block.step}):",
            *body,
        )
    if target is ControlTarget.FORTRAN:
        # A Fortran do-loop bound is inclusive, so the exclusive stop used by
        # every other target becomes stop - 1.
        return (
            f"do {block.induction} = {block.start}, "
            f"({block.stop}) - 1, {block.step}",
            *body,
            "end do",
        )
    declaration = "int " if target in {ControlTarget.C, ControlTarget.GLSL} else ""
    return (
        f"for ({declaration}{block.induction} = {block.start}; "
        f"{block.induction} < {block.stop}; "
        f"{block.induction} += {block.step}) {{",
        *body,
        "}",
    )


def _predicate_spelling(value_id: int, target: ControlTarget) -> str:
    value = f"value_{int(value_id)}"
    return f"bool({value})" if target is ControlTarget.PYTHON else value


def _render_expression(
    expression: ControlExpression,
    target: ControlTarget,
) -> str:
    if expression.op == "value":
        return f"value_{int(expression.value_id)}"
    if expression.op == "const":
        return repr(expression.literal).lower() if target is not ControlTarget.PYTHON else repr(expression.literal)
    if expression.op in {"item", "float", "int", "bool"}:
        return _render_expression(expression.operands[0], target)
    unary = {"not": "!", "neg": "-"}
    if expression.op in unary:
        token = "not " if target is ControlTarget.PYTHON and expression.op == "not" else unary[expression.op]
        return f"({token}{_render_expression(expression.operands[0], target)})"
    binary = {
        "add": "+", "sub": "-", "mul": "*", "div": "/",
        "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
        "eq": "==", "ne": "!=",
        "and": "and" if target is ControlTarget.PYTHON else "&&",
        "or": "or" if target is ControlTarget.PYTHON else "||",
    }[expression.op]
    return f"({_render_expression(expression.operands[0], target)} {binary} {_render_expression(expression.operands[1], target)})"


def _render_while(block: WhileBlock, target: ControlTarget) -> tuple[str, ...]:
    condition = render_control_block(block.condition, target)
    body = render_control_block(block.body, target)
    predicate = (
        _render_expression(block.predicate_expression, target)
        if block.predicate_expression is not None
        else _predicate_spelling(block.predicate_value_id, target)
    )
    if target is ControlTarget.PYTHON:
        return (
            *condition,
            f"while {predicate}:",
            *_indent((*body, *condition)),
        )
    if target is ControlTarget.FORTRAN:
        return (
            *condition,
            f"do while ({predicate})",
            *_indent((*body, *condition)),
            "end do",
        )
    return (
        *condition,
        f"while ({predicate}) {{",
        *_indent((*body, *condition)),
        "}",
    )


def _render_tick(
    block: StateMachineTick,
    target: ControlTarget,
) -> tuple[str, ...]:
    if target is ControlTarget.PYTHON:
        lines = []
        for index, (value, body) in enumerate(block.cases):
            keyword = "if" if index == 0 else "elif"
            lines.append(f"{keyword} {block.state} == {value}:")
            lines.extend(_indent(render_control_block(body, target)))
        if block.default is not None:
            lines.append("else:")
            lines.extend(_indent(render_control_block(block.default, target)))
        return tuple(lines)
    if target is ControlTarget.FORTRAN:
        lines = [f"select case ({block.state})"]
        for value, body in block.cases:
            lines.append(f"case ({value})")
            lines.extend(_indent(render_control_block(body, target)))
        if block.default is not None:
            lines.append("case default")
            lines.extend(_indent(render_control_block(block.default, target)))
        lines.append("end select")
        return tuple(lines)
    lines = [f"switch ({block.state}) {{"]
    for value, body in block.cases:
        lines.append(f"    case {value}:")
        lines.extend(_indent(render_control_block(body, target), 8))
        lines.append("        break;")
    if block.default is not None:
        lines.append("    default:")
        lines.extend(_indent(render_control_block(block.default, target), 8))
        lines.append("        break;")
    lines.append("}")
    return tuple(lines)


def render_control_block(
    block: ControlBlock,
    target: ControlTarget,
) -> tuple[str, ...]:
    """Render control syntax without making any scheduling decisions."""

    if isinstance(block, StatementBlock):
        return block.lines
    if isinstance(block, SequenceBlock):
        return tuple(
            line
            for child in block.blocks
            for line in render_control_block(child, target)
        )
    if isinstance(block, LoopBlock):
        return _render_loop(block, target)
    if isinstance(block, WhileBlock):
        return _render_while(block, target)
    if isinstance(block, LoopControlBlock):
        statement = (
            f"{block.action}"
            if target in {ControlTarget.PYTHON, ControlTarget.FORTRAN}
            else f"{block.action};"
        )
        if block.predicate_value_id is None:
            return (statement,)
        predicate = (
            _render_expression(block.predicate_expression, target)
            if block.predicate_expression is not None
            else _predicate_spelling(block.predicate_value_id, target)
        )
        if not block.expect_true:
            predicate = (
                f"not {predicate}"
                if target is ControlTarget.PYTHON
                else f"!({predicate})"
            )
        if target is ControlTarget.PYTHON:
            return (f"if {predicate}:", f"    {statement}")
        if target is ControlTarget.FORTRAN:
            return (f"if ({predicate}) then", f"    {statement}", "end if")
        return (f"if ({predicate}) {{", f"    {statement}", "}")
    if isinstance(block, StateMachineTick):
        return _render_tick(block, target)
    if isinstance(block, ParallelDeployment):
        if target is ControlTarget.PYTHON:
            raise ValueError(
                "parallel deployment has no implicit Python execution; "
                "the planner must select a concrete parallel runtime"
            )
        # Lanes are independent, but their statements may share one compiled
        # program.  Rendering them consecutively preserves each lane's local
        # order without inventing dependencies between lanes.
        return tuple(
            line
            for lane in block.lanes
            for line in render_control_block(lane, target)
        )
    if isinstance(block, CallBlock):
        # Value IDs have already been unified by the hierarchy planner.
        # A target renderer therefore sees the callee as ordinary nested
        # control, not as a host-language function call.
        return render_control_block(block.callee, target)
    if isinstance(block, ValidationBlock):
        predicate = f"value_{int(block.predicate_value_id)}"
        expected = "true" if block.expect_true else "false"
        if target is ControlTarget.PYTHON:
            return (
                f"if bool({predicate}) is not {expected}:",
                f"    raise RuntimeError('validation {block.error_code}')",
            )
        return (
            f"if (bool({predicate}) != {expected}) {{",
            f"    turing_validation_error({int(block.error_code)}u);",
            "}",
        )
    if isinstance(block, StreamPublishBlock):
        count = (
            "-1"
            if block.count_value_id is None
            else f"value_{int(block.count_value_id)}"
        )
        predicate = (
            "true"
            if block.predicate_value_id is None
            else f"bool(value_{int(block.predicate_value_id)})"
        )
        final = "true" if block.final else "false"
        return (
            f"if ({predicate}) {{",
            f"    turing_stream_publish({int(block.stream_id)}u, "
            f"value_{int(block.value_id)}, {count}, {final});",
            "}",
        )
    raise TypeError(f"unknown control block {type(block).__name__}")


def _region_marker(block: StatementBlock) -> int | None:
    if len(block.lines) != 1:
        return None
    marker = block.lines[0]
    prefix = "__scheduled_region_"
    if not marker.startswith(prefix) or not marker.endswith("__"):
        return None
    return int(marker[len(prefix):-2])


def compose_region_code(
    program: ControlProgram,
    target: ControlTarget,
    regions: Iterable[RegionCode],
) -> ControlProgram:
    """Substitute late-selected interiors into one shell control program.

    Device GLSL shells remain homogeneous.  A C shell may orchestrate a
    separately compiled GLSL interior when selection supplied an explicit C
    launch block.  Other cross-language combinations require their own
    deliberately defined capability and are rejected here.
    """

    selected = tuple(regions)
    by_region = {region.region_index: region for region in selected}
    if len(by_region) != len(selected):
        raise ValueError("a shell cannot select two implementations of one region")
    expected = tuple(program.region_indices)
    if set(by_region) != set(expected):
        raise ValueError(
            "selected code clusters do not match the logical shell: "
            f"expected={expected!r}, selected={tuple(by_region)!r}"
        )
    def permitted(region: RegionCode) -> bool:
        if region.target is target:
            return True
        return (
            target is ControlTarget.C
            and region.target is ControlTarget.GLSL
            and region.launch_body is not None
        )

    wrong = tuple(
        region.region_index for region in selected if not permitted(region)
    )
    if wrong:
        if target is ControlTarget.GLSL:
            reason = "glsl shell requires glsl interiors"
        elif target is ControlTarget.C:
            reason = (
                "c shell permits C interiors or GLSL interiors with an "
                "explicit C launch block and OpenGL context"
            )
        else:
            reason = "python shell requires python interiors"
        raise ValueError(
            f"{reason}; incompatible regions={wrong!r}"
        )
    consumed = []

    def substitute(block: ControlBlock) -> ControlBlock:
        if isinstance(block, StatementBlock):
            region_index = _region_marker(block)
            if region_index is None:
                return block
            consumed.append(region_index)
            region = by_region[region_index]
            if region.target is target:
                return region.body
            # The foreign payload is compiled independently.  Only its
            # explicit ABI-level invocation belongs in this host shell.
            assert region.launch_body is not None
            return region.launch_body
        if isinstance(block, SequenceBlock):
            return SequenceBlock(tuple(substitute(child) for child in block.blocks))
        if isinstance(block, LoopBlock):
            return LoopBlock(
                block.induction,
                block.start,
                block.stop,
                block.step,
                substitute(block.body),
                block.carried_aliases,
                block.parallel_iterations,
                block.dispatch_shell,
                block.recursion_region_id,
                block.schedule_preference,
            )
        if isinstance(block, WhileBlock):
            return WhileBlock(
                block.predicate_value_id,
                substitute(block.condition),
                substitute(block.body),
                block.carried_aliases,
                block.recursion_region_id,
                block.predicate_expression,
            )
        if isinstance(block, LoopControlBlock):
            return block
        if isinstance(block, StateMachineTick):
            return StateMachineTick(
                block.state,
                tuple((value, substitute(body)) for value, body in block.cases),
                None if block.default is None else substitute(block.default),
            )
        if isinstance(block, ParallelDeployment):
            return ParallelDeployment(
                tuple(substitute(lane) for lane in block.lanes),
                block.schedule_preference,
            )
        if isinstance(block, CallBlock):
            return CallBlock(
                block.callsite_id,
                substitute(block.callee),
                block.argument_bindings,
                block.result_bindings,
            )
        if isinstance(block, ValidationBlock):
            if (
                retained_values is not None
                and int(block.predicate_value_id) not in retained_values
            ):
                return None
            return block
        if isinstance(block, StreamPublishBlock):
            return block
        raise TypeError(f"unknown control block {type(block).__name__}")

    root = substitute(program.root)
    if tuple(consumed) != expected:
        raise ValueError(
            "logical shell must consume each region exactly once: "
            f"expected={expected!r}, consumed={tuple(consumed)!r}"
        )
    return ControlProgram(
        root=root,
        region_indices=expected,
        uniforms=program.uniforms,
        value_aliases=program.value_aliases,
        iterable_bindings=program.iterable_bindings,
        static_iterable_bindings=program.static_iterable_bindings,
        collection_bindings=program.collection_bindings,
        closure_iterable_bindings=program.closure_iterable_bindings,
        recursion_regions=program.recursion_regions,
        deployment_regions=program.deployment_regions,
    )


def project_control_regions(
    program: ControlProgram,
    retained_region_indices: Iterable[int],
    *,
    retained_value_ids: Iterable[int] | None = None,
) -> ControlProgram:
    """Project compiled control onto regions that still require runtime work.

    Planning happens before static/structural regions are resolved.  Once such
    a region disappears, its marker must be removed recursively from loops,
    branches and nested closures rather than forcing a fabricated backend
    operation.  A control construct whose runtime body becomes empty collapses
    with it.
    """

    retained = frozenset(int(index) for index in retained_region_indices)
    retained_values = (
        None
        if retained_value_ids is None
        else frozenset(int(value) for value in retained_value_ids)
    )

    def project(block: ControlBlock) -> ControlBlock | None:
        if isinstance(block, StatementBlock):
            marker = _region_marker(block)
            if marker is not None and marker not in retained:
                return None
            return block
        if isinstance(block, SequenceBlock):
            children = tuple(
                projected
                for child in block.blocks
                if (projected := project(child)) is not None
            )
            return SequenceBlock(children) if children else None
        if isinstance(block, LoopBlock):
            body = project(block.body)
            if body is None:
                return None
            return LoopBlock(
                block.induction,
                block.start,
                block.stop,
                block.step,
                body,
                tuple(
                    (updated, initial)
                    for updated, initial in block.carried_aliases
                    if retained_values is None
                    or (
                        int(updated) in retained_values
                        and int(initial) in retained_values
                    )
                ),
                block.parallel_iterations,
                block.dispatch_shell,
                block.recursion_region_id,
                block.schedule_preference,
            )
        if isinstance(block, WhileBlock):
            condition = project(block.condition)
            body = project(block.body)
            if body is None or (
                condition is None and block.predicate_expression is None
            ):
                return None
            return WhileBlock(
                block.predicate_value_id,
                condition or SequenceBlock(()),
                body,
                tuple(
                    (updated, initial)
                    for updated, initial in block.carried_aliases
                    if retained_values is None
                    or (
                        int(updated) in retained_values
                        and int(initial) in retained_values
                    )
                ),
                block.recursion_region_id,
                block.predicate_expression,
            )
        if isinstance(block, LoopControlBlock):
            if (
                block.predicate_value_id is not None
                and retained_values is not None
                and int(block.predicate_value_id) not in retained_values
            ):
                return None
            return block
        if isinstance(block, StateMachineTick):
            cases = tuple(
                (value, projected)
                for value, body in block.cases
                if (projected := project(body)) is not None
            )
            default = (
                None if block.default is None else project(block.default)
            )
            return (
                StateMachineTick(block.state, cases, default)
                if cases or default is not None else None
            )
        if isinstance(block, ParallelDeployment):
            lanes = tuple(
                projected
                for lane in block.lanes
                if (projected := project(lane)) is not None
            )
            return (
                ParallelDeployment(lanes, block.schedule_preference)
                if lanes else None
            )
        if isinstance(block, CallBlock):
            callee = project(block.callee)
            if callee is None:
                return None
            return CallBlock(
                block.callsite_id,
                callee,
                block.argument_bindings,
                block.result_bindings,
            )
        if isinstance(block, ValidationBlock):
            return block
        if isinstance(block, StreamPublishBlock):
            return block
        raise TypeError(f"unknown control block {type(block).__name__}")

    root = project(program.root) or SequenceBlock(())

    active_inductions: set[str] = set()
    active_recursion_regions: set[int] = set()

    def gather_inductions(block: ControlBlock) -> None:
        if isinstance(block, LoopBlock):
            active_inductions.add(str(block.induction))
            if block.recursion_region_id is not None:
                active_recursion_regions.add(int(block.recursion_region_id))
            gather_inductions(block.body)
        elif isinstance(block, WhileBlock):
            if block.recursion_region_id is not None:
                active_recursion_regions.add(int(block.recursion_region_id))
            gather_inductions(block.condition)
            gather_inductions(block.body)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                gather_inductions(child)
        elif isinstance(block, StateMachineTick):
            for _value, body in block.cases:
                gather_inductions(body)
            if block.default is not None:
                gather_inductions(block.default)
        elif isinstance(block, ParallelDeployment):
            for lane in block.lanes:
                gather_inductions(lane)
        elif isinstance(block, CallBlock):
            gather_inductions(block.callee)

    gather_inductions(root)
    projected_deployments = []
    for deployment in program.deployment_regions:
        lanes = tuple(
            replace(
                lane,
                region_indices=tuple(
                    index
                    for index in lane.region_indices
                    if int(index) in retained
                ),
                value_ids=tuple(
                    value_id
                    for value_id in lane.value_ids
                    if retained_values is None
                    or int(value_id) in retained_values
                ),
            )
            for lane in deployment.lanes
        )
        lanes = tuple(
            projected_lane
            for source_lane, projected_lane in zip(
                deployment.lanes, lanes
            )
            if (
                projected_lane.region_indices
                or (
                    not source_lane.region_indices
                    and (
                        projected_lane.value_ids
                        or projected_lane.source_node_ids
                    )
                )
            )
        )
        lanes = tuple(
            replace(lane, index=index)
            for index, lane in enumerate(lanes)
        )
        if lanes:
            projected_deployments.append(replace(deployment, lanes=lanes))
    return ControlProgram(
        root,
        tuple(
            region_index
            for region_index in program.region_indices
            if region_index in retained
        ),
        program.uniforms,
        tuple(
            (updated, initial)
            for updated, initial in program.value_aliases
            if retained_values is None
            or (
                int(updated) in retained_values
                and int(initial) in retained_values
            )
        ),
        tuple(
            binding
            for binding in program.iterable_bindings
            if str(binding[2]) in active_inductions
        ),
        tuple(
            binding
            for binding in program.static_iterable_bindings
            if str(binding[2]) in active_inductions
        ),
        tuple(
            binding
            for binding in program.collection_bindings
            if str(binding[2]) in active_inductions
        ),
        tuple(
            binding
            for binding in program.closure_iterable_bindings
            if str(binding[2]) in active_inductions
        ),
        tuple(
            region
            for region in program.recursion_regions
            if region.region_id in active_recursion_regions
        ),
        tuple(projected_deployments),
    )


def overlay_scheduled_control(
    region_indices: Iterable[int],
    controls: Iterable[ControlProgram],
) -> ControlProgram:
    """Overlay planned control blocks on the flat scheduled region order."""

    order = tuple(int(index) for index in region_indices)
    positions = {region_index: index for index, region_index in enumerate(order)}
    if len(positions) != len(order):
        raise ValueError("scheduled region order contains duplicates")
    replacements = {}
    covered = set()
    uniforms = []
    aliases = []
    iterable_bindings = []
    static_iterable_bindings = []
    collection_bindings = []
    closure_iterable_bindings = []
    recursion_regions = []
    deployment_regions = []
    controls = tuple(controls)
    controlled_sets = tuple(
        frozenset(control.region_indices) for control in controls
    )

    def embed(
        block: ControlBlock,
        nested_root: ControlBlock,
        nested_regions: frozenset[int],
    ) -> tuple[ControlBlock | None, bool]:
        """Replace one nested region span with its planner-owned control."""

        if isinstance(block, StatementBlock):
            marker = _region_marker(block)
            if marker not in nested_regions:
                return block, False
            # Only the first lexical marker receives the nested control; the
            # remaining markers belong to that control and disappear from the
            # parent's flat body.
            return None, True
        if isinstance(block, SequenceBlock):
            children = []
            inserted = False
            for child in block.blocks:
                projected, consumed = embed(
                    child, nested_root, nested_regions
                )
                if consumed and not inserted:
                    children.append(nested_root)
                    inserted = True
                if projected is not None:
                    children.append(projected)
            return SequenceBlock(tuple(children)), inserted
        if isinstance(block, LoopBlock):
            body, consumed = embed(
                block.body, nested_root, nested_regions
            )
            return (
                LoopBlock(
                    block.induction,
                    block.start,
                    block.stop,
                    block.step,
                    body or SequenceBlock(()),
                    block.carried_aliases,
                    block.parallel_iterations,
                    block.dispatch_shell,
                    block.recursion_region_id,
                    block.schedule_preference,
                ),
                consumed,
            )
        if isinstance(block, WhileBlock):
            condition, condition_consumed = embed(
                block.condition, nested_root, nested_regions
            )
            body, body_consumed = embed(
                block.body, nested_root, nested_regions
            )
            return (
                WhileBlock(
                    block.predicate_value_id,
                    condition or SequenceBlock(()),
                    body or SequenceBlock(()),
                    block.carried_aliases,
                    block.recursion_region_id,
                    block.predicate_expression,
                ),
                condition_consumed or body_consumed,
            )
        if isinstance(block, LoopControlBlock):
            return block, False
        if isinstance(block, StateMachineTick):
            cases = []
            consumed_any = False
            for value, body in block.cases:
                projected, consumed = embed(
                    body, nested_root, nested_regions
                )
                cases.append((value, projected or SequenceBlock(())))
                consumed_any |= consumed
            default = None
            if block.default is not None:
                default, consumed = embed(
                    block.default, nested_root, nested_regions
                )
                default = default or SequenceBlock(())
                consumed_any |= consumed
            return (
                StateMachineTick(block.state, tuple(cases), default),
                consumed_any,
            )
        if isinstance(block, ParallelDeployment):
            lanes = []
            consumed_any = False
            for lane in block.lanes:
                projected, consumed = embed(
                    lane, nested_root, nested_regions
                )
                lanes.append(projected or SequenceBlock(()))
                consumed_any |= consumed
            return (
                ParallelDeployment(tuple(lanes), block.schedule_preference),
                consumed_any,
            )
        if isinstance(block, CallBlock):
            callee, consumed = embed(
                block.callee, nested_root, nested_regions
            )
            return (
                CallBlock(
                    block.callsite_id,
                    callee or SequenceBlock(()),
                    block.argument_bindings,
                    block.result_bindings,
                ),
                consumed,
            )
        return block, False

    nested_roots: dict[int, ControlBlock] = {}

    def nested_root(index: int, visiting=frozenset()) -> ControlBlock:
        if index in nested_roots:
            return nested_roots[index]
        if index in visiting:
            raise ValueError("cyclic loop-control containment")
        root = controls[index].root
        candidates = [
            child
            for child, child_regions in enumerate(controlled_sets)
            if child != index
            and child_regions
            and child_regions < controlled_sets[index]
            and not any(
                child_regions < middle_regions < controlled_sets[index]
                for middle, middle_regions in enumerate(controlled_sets)
                if middle not in {index, child}
            )
        ]
        for child in sorted(
            candidates,
            key=lambda item: min(
                positions[region] for region in controlled_sets[item]
            ),
        ):
            root, consumed = embed(
                root,
                nested_root(child, visiting | {index}),
                controlled_sets[child],
            )
            if not consumed:
                raise ValueError(
                    "nested control regions are absent from parent body: "
                    f"parent={tuple(controls[index].region_indices)!r}, "
                    f"child={tuple(controls[child].region_indices)!r}"
                )
        nested_roots[index] = root
        return root

    maximal = [
        index
        for index, regions in enumerate(controlled_sets)
        if regions and not any(
            regions < other
            for other in controlled_sets
        )
    ]
    for index, control in enumerate(controls):
        controlled = tuple(control.region_indices)
        if not controlled:
            continue
        missing = set(controlled) - set(order)
        if missing:
            raise ValueError(
                "control overlay does not partition the schedule: "
                f"missing={sorted(missing)!r}"
            )
        uniforms.extend(control.uniforms)
        aliases.extend(control.value_aliases)
        iterable_bindings.extend(control.iterable_bindings)
        static_iterable_bindings.extend(control.static_iterable_bindings)
        collection_bindings.extend(control.collection_bindings)
        closure_iterable_bindings.extend(
            control.closure_iterable_bindings
        )
        recursion_regions.extend(control.recursion_regions)
        deployment_base = len(deployment_regions)
        deployment_regions.extend(
            replace(region, region_id=deployment_base + offset)
            for offset, region in enumerate(control.deployment_regions)
        )
        if index not in maximal:
            continue
        overlap = set(controlled) & covered
        if overlap:
            raise ValueError(
                "maximal control blocks overlap without containment: "
                f"overlap={sorted(overlap)!r}"
            )
        first = min(controlled, key=positions.__getitem__)
        replacements[first] = nested_root(index)
        covered.update(controlled)
    blocks = []
    for region_index in order:
        replacement = replacements.get(region_index)
        if replacement is not None:
            blocks.append(replacement)
        elif region_index not in covered:
            blocks.append(StatementBlock((
                f"__scheduled_region_{region_index}__",
            )))
    return ControlProgram(
        root=SequenceBlock(tuple(blocks)),
        region_indices=order,
        uniforms=tuple(dict.fromkeys(uniforms)),
        value_aliases=tuple(dict.fromkeys(aliases)),
        iterable_bindings=tuple(dict.fromkeys(iterable_bindings)),
        static_iterable_bindings=tuple(
            dict.fromkeys(static_iterable_bindings)
        ),
        collection_bindings=tuple(dict.fromkeys(collection_bindings)),
        closure_iterable_bindings=tuple(
            dict.fromkeys(closure_iterable_bindings)
        ),
        recursion_regions=tuple(dict.fromkeys(recursion_regions)),
        deployment_regions=tuple(deployment_regions),
    )


def render_control_program(
    program: ControlProgram,
    target: ControlTarget,
) -> str:
    return "\n".join(render_control_block(program.root, target))


def render_c_shell(
    program: ControlProgram,
    regions: Iterable[RegionCode],
    *,
    function_name: str,
    parameters: Iterable[str] = (),
    return_type: str = "void",
) -> str:
    """Finalize a logical shell as one C function after cluster selection."""

    if not function_name.isidentifier():
        raise ValueError(f"invalid C shell name {function_name!r}")
    composed = compose_region_code(program, ControlTarget.C, regions)
    body = _indent(render_control_block(composed.root, ControlTarget.C))
    return "\n".join((
        f"{return_type} {function_name}({', '.join(parameters) or 'void'}) {{",
        *body,
        "}",
        "",
    ))


def render_python_shell(
    program: ControlProgram,
    regions: Iterable[RegionCode],
    *,
    function_name: str,
    parameters: Iterable[str] = (),
) -> str:
    """Finalize a logical shell as Python source after cluster selection."""

    if not function_name.isidentifier():
        raise ValueError(f"invalid Python shell name {function_name!r}")
    composed = compose_region_code(program, ControlTarget.PYTHON, regions)
    body = _indent(render_control_block(composed.root, ControlTarget.PYTHON))
    if not body:
        body = ("    pass",)
    return "\n".join((
        f"def {function_name}({', '.join(parameters)}):",
        *body,
        "",
    ))


def compile_python_shell(
    program: ControlProgram,
    regions: Iterable[RegionCode],
    *,
    function_name: str,
    parameters: Iterable[str] = (),
    namespace: Mapping[str, Any] | None = None,
    abstract_tensor_backend: str | None = None,
):
    """Finalize and load a Python callable from the logical shell."""

    source = render_python_shell(
        program,
        regions,
        function_name=function_name,
        parameters=parameters,
    )
    if abstract_tensor_backend is not None:
        from src.common.tensors.abstraction import AbstractTensor

        lines = source.splitlines()
        source = "\n".join((
            lines[0],
            f"    with AbstractTensor.use_backend({abstract_tensor_backend!r}):",
            *("    " + line for line in lines[1:] if line),
            "",
        ))
    scope = dict(namespace or {})
    if abstract_tensor_backend is not None:
        scope["AbstractTensor"] = AbstractTensor
    exec(compile(source, f"<compiled-shell:{function_name}>", "exec"), scope)
    result = scope[function_name]
    result.__compiled_shell_source__ = source
    return result


@dataclass(frozen=True)
class CFFICallable:
    """Keep the CFFI owner and library alive around one compiled function."""

    ffi: Any
    library: Any
    function: Any
    source: str

    def __call__(self, *args):
        return self.function(*args)


def compile_cffi_shell(
    program: ControlProgram,
    regions: Iterable[RegionCode],
    *,
    function_name: str,
    parameters: Iterable[str],
    c_declaration: str,
    return_type: str = "void",
    preamble: str = "",
    extra_compile_args: Iterable[str] = (),
) -> CFFICallable:
    """Finalize one C shell and expose it as an owning CFFI callable."""

    from cffi import FFI

    function_source = render_c_shell(
        program,
        regions,
        function_name=function_name,
        parameters=parameters,
        return_type=return_type,
    )
    source = "\n".join(part for part in (preamble, function_source) if part)
    ffi = FFI()
    ffi.cdef(c_declaration)
    library = ffi.verify(
        source,
        extra_compile_args=list(extra_compile_args),
    )
    return CFFICallable(
        ffi=ffi,
        library=library,
        function=getattr(library, function_name),
        source=source,
    )


__all__ = [
    "ControlBlock",
    "ControlDeploymentLane",
    "ControlDeploymentRegion",
    "ControlExpression",
    "CallBlock",
    "ValidationBlock",
    "ControlProgram",
    "ControlTarget",
    "ControlUniform",
    "CFFICallable",
    "LoopBlock",
    "LoopControlBlock",
    "ParallelDeployment",
    "RecursionRegion",
    "RegionCode",
    "SequenceBlock",
    "StateMachineTick",
    "StatementBlock",
    "StreamPublishBlock",
    "WhileBlock",
    "compile_cffi_shell",
    "compile_python_shell",
    "compose_region_code",
    "overlay_scheduled_control",
    "project_control_regions",
    "render_c_shell",
    "render_control_block",
    "render_control_program",
    "render_python_shell",
]
