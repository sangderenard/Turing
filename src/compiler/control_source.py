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
class ConditionalBlock:
    """Execute one compiled control body under a resident predicate."""

    predicate_value_id: int
    body: "ControlBlock"
    orelse: "ControlBlock | None" = None
    expect_true: bool = True
    predicate_expression: ControlExpression | None = None
    # (true-arm value, false-arm value, pre-branch value, merged value).
    # A missing arm repeats the pre-branch value and is encoded by giving that
    # arm the same id as ``initial_value_id``.
    carried_aliases: tuple[tuple[int, int, int, int], ...] = ()
    source_node_id: int | None = None
    # Resident sequences cannot be merged as scalar SSA values.  Each tuple
    # has the same (true, false, initial, merged) shape as ``carried_aliases``;
    # lowering replaces the initial arena from the selected branch and keeps
    # the merged spelling correlated with that one storage descriptor.
    carried_sequence_aliases: tuple[tuple[int, int, int, int], ...] = ()
    # (origin sequence, selected row handle, result value, physical column,
    # dtype).  These loads are valid only in the selected arm and therefore
    # belong at conditional entry, never beside the optional query itself.
    entry_record_projections: tuple[
        tuple[int, int, int, int, str], ...
    ] = ()


@dataclass(frozen=True)
class ControlSequenceMutation:
    """One explicit mutation of caller-provided sequence/table storage."""

    sequence_value_id: int
    operator: str
    argument_value_ids: tuple[int, ...]
    effect_node_id: int
    policy: str | None = None
    argument_kind: str = "value"
    predicate_expression: ControlExpression | None = None
    # Expressions aligned with ``argument_value_ids``.  A non-None entry
    # proves how a coordinator scalar is computed inside the selected arm,
    # preventing its result ID from becoming an invented function argument.
    argument_expressions: tuple[ControlExpression | None, ...] = ()
    extraction_identity: str | None = None


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
    # ``(port id, initial id, updated id)`` per carried binding: the
    # LoopResult port is the value id every post-loop consumer was rewired
    # to, so the SSA lowering must define it as the carried Phi's exit value.
    # Left uncarried, each port materialized as a producerless argument and
    # every reduction result after the loop read its own seed.
    result_ports: tuple[tuple[int, int, int], ...] = ()
    #: ``(initial id, literal)`` for carried seeds that folded to constants;
    #: the SSA lowering materializes these as Const instead of inventing a
    #: producerless argument for an evaporated node.
    carried_seeds: tuple[tuple[int, float], ...] = ()
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
    sequence_mutations: tuple[ControlSequenceMutation, ...] = ()
    # Forward ranges normally use ``lt``. Reverse schedules use ``gt`` with a
    # negative step; keeping this explicit prevents C/SSA renderers from
    # silently applying a forward-only comparison to an adjoint loop.
    comparison: str = "lt"
    # Terminal source exits run after resident effects for the iteration.  A
    # planner may populate this only for a return proven to be in tail
    # position; ordinary break/continue remain lexically embedded in body.
    terminal_controls: tuple["LoopControlBlock", ...] = ()
    # Exact authored ProcessGraph loop identity.  Like WhileBlock's identity,
    # this survives scheduling so verification can prove that source loops
    # became CFG loops rather than merely observing generic Phi/Lt blocks.
    source_loop_node_id: int | None = None

    def __post_init__(self) -> None:
        preference = str(self.schedule_preference).lower()
        if preference not in {"asap", "alap"}:
            raise ValueError(
                "loop schedule preference must be 'asap' or 'alap'"
            )
        object.__setattr__(self, "schedule_preference", preference)
        comparison = str(self.comparison).lower()
        if comparison not in {"lt", "gt"}:
            raise ValueError("loop comparison must be 'lt' or 'gt'")
        object.__setattr__(self, "comparison", comparison)


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
    result_ports: tuple[tuple[int, int, int], ...] = ()
    carried_seeds: tuple[tuple[int, float], ...] = ()
    recursion_region_id: int | None = None
    predicate_expression: ControlExpression | None = None
    sequence_mutations: tuple[ControlSequenceMutation, ...] = ()
    # Exact source ProcessGraph loop identity.  A source-linked call retains
    # this same identity in ``PlanCall.enclosing_loop_ids``; carrying it into
    # Control IR lets the SSA linker place that call in the authored while
    # body instead of guessing from a later result consumer.
    source_loop_node_id: int | None = None
    terminal_controls: tuple["LoopControlBlock", ...] = ()


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
    source_action: str | None = None

    def __post_init__(self) -> None:
        if self.action not in {"break", "continue"}:
            raise ValueError("loop control action must be break or continue")
        if self.source_action not in {None, "break", "continue", "loop-return"}:
            raise ValueError("unknown source loop-control action")


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
class ExternalReferenceCallBlock:
    """One authored call through the shell external-reference capability."""

    callsite_id: int
    identity: str
    argument_value_ids: tuple[int, ...]
    keyword_argument_value_ids: tuple[tuple[str, int], ...]
    result_value_id: int
    result_dtype: str
    shell_abi: str = "turing-shell-io-abi.external_references"
    external_domain: str = "host_system"
    native_abi: str = ""
    runtime_owner: str = ""
    shell_profiles: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationBlock:
    """Device-side predicate whose failure is reported through shell errors."""

    predicate_value_id: int
    error_code: int
    expect_true: bool = True
    predicate_expression: ControlExpression | None = None
    extraction_identity: str | None = None


@dataclass(frozen=True)
class SequenceMutationBlock:
    """One lexical resident-sequence effect outside implicit loop storage."""

    mutation: ControlSequenceMutation


@dataclass(frozen=True)
class SequenceQueryBlock:
    """Read one scalar fact from a resident sequence at lexical position."""

    result_value_id: int
    sequence_value_id: int
    operation: str
    default_value_id: int | None = None
    source_call_node_id: int | None = None
    extraction_identity: str | None = None
    result_alias_ids: tuple[int, ...] = ()
    producer_loop_node_id: int | None = None
    # A first-or-default query over a derived record sequence yields a tagged
    # integer row handle.  The default arm uses -1 and subsequent source
    # ``is [not] None`` tests consume that tag.
    row_handle: bool = False

    def __post_init__(self) -> None:
        if self.operation not in {"length", "first_or_default"}:
            raise ValueError("unknown resident sequence query")
        if self.operation == "first_or_default" and self.default_value_id is None:
            raise ValueError("first_or_default requires an explicit default")


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
    | ConditionalBlock
    | LoopBlock
    | WhileBlock
    | LoopControlBlock
    | StateMachineTick
    | ParallelDeployment
    | CallBlock
    | ExternalReferenceCallBlock
    | ValidationBlock
    | SequenceMutationBlock
    | SequenceQueryBlock
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
    # (resident iterable value id, target value id, induction, projection).
    # Projection is ``"induction"`` for enumerate's counter, ``None`` for
    # the whole resident element, or a zero-based integer field within a
    # destructured resident tuple/row.
    projected_iterable_bindings: tuple[
        tuple[int, int, str, object], ...
    ] = ()
    # Source ``if`` nodes whose complete arm effects are represented by a
    # specialized control form (for example, predicated resident mutations
    # plus a loop-return edge), rather than by a ConditionalBlock.  These are
    # semantic-accounting identities, not permission to omit either arm.
    specialized_conditional_node_ids: tuple[int, ...] = ()


def control_dependency_value_ids(control: ControlProgram | None) -> frozenset[int]:
    """Return every value identity explicitly consumed by planned control."""

    values: set[int] = set()
    if control is None:
        return frozenset()

    values.update(int(uniform.value_id) for uniform in control.uniforms)
    for bindings in (
        control.value_aliases,
        control.iterable_bindings,
        control.static_iterable_bindings,
        control.collection_bindings,
        control.closure_iterable_bindings,
        control.projected_iterable_bindings,
    ):
        for binding in bindings:
            for value in binding:
                if isinstance(value, int):
                    values.add(int(value))
                elif isinstance(value, tuple):
                    values.update(int(item) for item in value if isinstance(item, int))

    def expression_values(expression: ControlExpression | None) -> None:
        if expression is None:
            return
        if expression.value_id is not None:
            values.add(int(expression.value_id))
        for operand in expression.operands:
            expression_values(operand)

    def carried_values(bindings) -> None:
        values.update(int(value_id) for binding in bindings for value_id in binding)

    def mutation_values(mutations) -> None:
        for mutation in mutations:
            values.add(int(mutation.sequence_value_id))
            values.update(int(value_id) for value_id in mutation.argument_value_ids)
            expression_values(mutation.predicate_expression)
            for expression in mutation.argument_expressions:
                expression_values(expression)

    def visit(block: ControlBlock) -> None:
        if isinstance(block, ValidationBlock):
            values.add(int(block.predicate_value_id))
            expression_values(block.predicate_expression)
        elif isinstance(block, SequenceMutationBlock):
            mutation_values((block.mutation,))
        elif isinstance(block, SequenceQueryBlock):
            values.add(int(block.result_value_id))
            values.update(int(value_id) for value_id in block.result_alias_ids)
            values.add(int(block.sequence_value_id))
            if block.default_value_id is not None:
                values.add(int(block.default_value_id))
        elif isinstance(block, StreamPublishBlock):
            values.add(int(block.value_id))
            if block.count_value_id is not None:
                values.add(int(block.count_value_id))
            if block.predicate_value_id is not None:
                values.add(int(block.predicate_value_id))
        elif isinstance(block, ExternalReferenceCallBlock):
            values.update(int(value_id) for value_id in block.argument_value_ids)
            values.update(
                int(value_id)
                for _name, value_id in block.keyword_argument_value_ids
            )
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                visit(child)
        elif isinstance(block, ConditionalBlock):
            values.add(int(block.predicate_value_id))
            expression_values(block.predicate_expression)
            carried_values(block.carried_aliases)
            carried_values(block.carried_sequence_aliases)
            visit(block.body)
            if block.orelse is not None:
                visit(block.orelse)
        elif isinstance(block, LoopBlock):
            carried_values(block.carried_aliases)
            mutation_values(block.sequence_mutations)
            visit(block.body)
            for terminal in block.terminal_controls:
                visit(terminal)
        elif isinstance(block, WhileBlock):
            values.add(int(block.predicate_value_id))
            expression_values(block.predicate_expression)
            carried_values(block.carried_aliases)
            mutation_values(block.sequence_mutations)
            visit(block.condition)
            visit(block.body)
            for terminal in block.terminal_controls:
                visit(terminal)
        elif isinstance(block, LoopControlBlock):
            if block.predicate_value_id is not None:
                values.add(int(block.predicate_value_id))
            expression_values(block.predicate_expression)
        elif isinstance(block, StateMachineTick):
            for _case, body in block.cases:
                visit(body)
            if block.default is not None:
                visit(block.default)
        elif isinstance(block, ParallelDeployment):
            for lane in block.lanes:
                visit(lane)
        elif isinstance(block, CallBlock):
            carried_values(block.argument_bindings)
            carried_values(block.result_bindings)
            visit(block.callee)

    visit(control.root)
    return frozenset(values)


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
    if block.sequence_mutations:
        raise ValueError(
            "sequence mutations require repository-SSA memory lowering; "
            "direct control-source rendering would hide their arena ABI"
        )
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
        exclusive_adjustment = "- 1" if block.comparison == "lt" else "+ 1"
        return (
            f"do {block.induction} = {block.start}, "
            f"({block.stop}) {exclusive_adjustment}, {block.step}",
            *body,
            "end do",
        )
    declaration = "int " if target in {ControlTarget.C, ControlTarget.GLSL} else ""
    comparison = "<" if block.comparison == "lt" else ">"
    return (
        f"for ({declaration}{block.induction} = {block.start}; "
        f"{block.induction} {comparison} {block.stop}; "
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
    if expression.op == "sequence_nonempty":
        raise ValueError(
            "sequence truth predicates require repository-SSA memory "
            "lowering; direct source rendering would hide the length-cell ABI"
        )
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
    if block.sequence_mutations:
        raise ValueError(
            "sequence mutations require repository-SSA memory lowering; "
            "direct control-source rendering would hide their arena ABI"
        )
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
    if isinstance(block, ConditionalBlock):
        predicate = (
            _render_expression(block.predicate_expression, target)
            if block.predicate_expression is not None
            else _predicate_spelling(block.predicate_value_id, target)
        )
        if not block.expect_true:
            predicate = (
                f"not {predicate}"
                if target is ControlTarget.PYTHON
                else f".not. ({predicate})"
                if target is ControlTarget.FORTRAN
                else f"!({predicate})"
            )
        body = render_control_block(block.body, target)
        orelse = (
            () if block.orelse is None
            else render_control_block(block.orelse, target)
        )
        if target is ControlTarget.PYTHON:
            return (
                f"if {predicate}:", *_indent(body, 4),
                *(("else:", *_indent(orelse, 4)) if orelse else ()),
            )
        if target is ControlTarget.FORTRAN:
            return (
                f"if ({predicate}) then", *_indent(body, 4),
                *(("else", *_indent(orelse, 4)) if orelse else ()),
                "end if",
            )
        return (
            f"if ({predicate}) {{", *_indent(body, 4),
            *(("} else {", *_indent(orelse, 4)) if orelse else ()),
            "}",
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
    if isinstance(block, ExternalReferenceCallBlock):
        if target is not ControlTarget.PYTHON:
            raise ValueError(
                "external-reference control requires a target-owned ABI adapter; "
                f"none is installed for {target.value} control rendering"
            )
        positional = ", ".join(
            f"value_{int(value_id)}" for value_id in block.argument_value_ids
        )
        positional_tuple = (
            "()" if not positional else
            f"({positional},)" if len(block.argument_value_ids) == 1 else
            f"({positional})"
        )
        keywords = ", ".join(
            f"{name!r}: value_{int(value_id)}"
            for name, value_id in block.keyword_argument_value_ids
        )
        return (
            f"value_{int(block.result_value_id)} = "
            f"__turing_external_call__({block.identity!r}, "
            f"{positional_tuple}, {{{keywords}}}, {block.result_dtype!r})",
        )
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
    if isinstance(block, SequenceMutationBlock):
        mutation = block.mutation
        return (
            f"turing_sequence_{mutation.operator}(value_{int(mutation.sequence_value_id)});",
        )
    if isinstance(block, SequenceQueryBlock):
        if block.operation == "length":
            return (
                f"value_{int(block.result_value_id)} = "
                f"turing_sequence_length(value_{int(block.sequence_value_id)});",
            )
        return (
            f"value_{int(block.result_value_id)} = "
            f"turing_sequence_first_or_default("
            f"value_{int(block.sequence_value_id)}, "
            f"value_{int(block.default_value_id)});",
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


def _marker_scope_paths(
    block: "ControlBlock",
    nested_regions: frozenset[int],
    path: tuple[str, ...] = ("top",),
) -> dict[int, tuple[str, ...]]:
    """Map each marker in ``nested_regions`` found under ``block`` to the
    scope-label path that owns it -- the granularity ``embed`` (inside
    ``overlay_scheduled_control``) inserts a nested control at.

    ``SequenceBlock`` is transparent: it does not open a new insertion
    scope, matching ``embed``'s own single local ``inserted`` flag per
    ``SequenceBlock``. Every composite construct's structural SLOT (a loop
    body, a while condition or body, a conditional arm, a state-machine
    case or default, a parallel lane) opens one, because ``embed`` recurses
    into each such slot with its own separate insertion decision. Two
    markers of the SAME nested control found at two different paths is
    exactly the shape ``embed`` cannot honor with one insertion -- calling
    this before ``embed`` runs turns that into a named refusal instead of
    a downstream duplicate-region crash.
    """

    found: dict[int, tuple[str, ...]] = {}
    if isinstance(block, StatementBlock):
        marker = _region_marker(block)
        if marker is not None and marker in nested_regions:
            found[marker] = path
        return found
    if isinstance(block, SequenceBlock):
        for child in block.blocks:
            found.update(_marker_scope_paths(child, nested_regions, path))
        return found
    if isinstance(block, ConditionalBlock):
        label = f"if(node={block.source_node_id})"
        found.update(_marker_scope_paths(
            block.body, nested_regions, path + (f"{label}.body",),
        ))
        if block.orelse is not None:
            found.update(_marker_scope_paths(
                block.orelse, nested_regions, path + (f"{label}.orelse",),
            ))
        return found
    if isinstance(block, LoopBlock):
        found.update(_marker_scope_paths(
            block.body, nested_regions,
            path + (f"loop({block.induction})",),
        ))
        return found
    if isinstance(block, WhileBlock):
        found.update(_marker_scope_paths(
            block.condition, nested_regions, path + ("while.condition",),
        ))
        found.update(_marker_scope_paths(
            block.body, nested_regions, path + ("while.body",),
        ))
        return found
    if isinstance(block, StateMachineTick):
        for value, body in block.cases:
            found.update(_marker_scope_paths(
                body, nested_regions, path + (f"case({value})",),
            ))
        if block.default is not None:
            found.update(_marker_scope_paths(
                block.default, nested_regions, path + ("default",),
            ))
        return found
    if isinstance(block, ParallelDeployment):
        for index, lane in enumerate(block.lanes):
            found.update(_marker_scope_paths(
                lane, nested_regions, path + (f"lane({index})",),
            ))
        return found
    if isinstance(block, CallBlock):
        # Lexical organization around nested control, not a runtime
        # scope of its own -- it has exactly one child, so it cannot
        # itself introduce a split the way a while's condition/body or a
        # conditional's two arms can.
        found.update(_marker_scope_paths(block.callee, nested_regions, path))
        return found
    return found


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
        if isinstance(block, ConditionalBlock):
            return ConditionalBlock(
                block.predicate_value_id,
                substitute(block.body),
                None if block.orelse is None else substitute(block.orelse),
                block.expect_true,
                block.predicate_expression,
                block.carried_aliases,
                block.source_node_id,
                block.carried_sequence_aliases,
            )
        if isinstance(block, LoopBlock):
            return LoopBlock(
                block.induction,
                block.start,
                block.stop,
                block.step,
                substitute(block.body),
                carried_aliases=block.carried_aliases,
                result_ports=block.result_ports,
                carried_seeds=block.carried_seeds,
                parallel_iterations=block.parallel_iterations,
                dispatch_shell=block.dispatch_shell,
                recursion_region_id=block.recursion_region_id,
                schedule_preference=block.schedule_preference,
                sequence_mutations=block.sequence_mutations,
                comparison=block.comparison,
                terminal_controls=block.terminal_controls,
                source_loop_node_id=block.source_loop_node_id,
            )
        if isinstance(block, WhileBlock):
            return WhileBlock(
                block.predicate_value_id,
                substitute(block.condition),
                substitute(block.body),
                carried_aliases=block.carried_aliases,
                result_ports=block.result_ports,
                carried_seeds=block.carried_seeds,
                recursion_region_id=block.recursion_region_id,
                predicate_expression=block.predicate_expression,
                sequence_mutations=block.sequence_mutations,
                source_loop_node_id=block.source_loop_node_id,
                terminal_controls=block.terminal_controls,
            )
        if isinstance(block, LoopControlBlock):
            return block
        if isinstance(block, SequenceMutationBlock):
            return block
        if isinstance(block, SequenceQueryBlock):
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
        if isinstance(block, ExternalReferenceCallBlock):
            return block
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
        projected_iterable_bindings=program.projected_iterable_bindings,
        specialized_conditional_node_ids=(
            program.specialized_conditional_node_ids
        ),
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
        if isinstance(block, ConditionalBlock):
            body = project(block.body)
            orelse = (
                None if block.orelse is None else project(block.orelse)
            )
            if body is None and orelse is None:
                return None
            return ConditionalBlock(
                block.predicate_value_id,
                body or SequenceBlock(()),
                orelse,
                block.expect_true,
                block.predicate_expression,
                tuple(
                    carried for carried in block.carried_aliases
                    if retained_values is None
                    or all(
                        int(value_id) in retained_values
                        for value_id in carried
                    )
                ),
                block.source_node_id,
                tuple(
                    carried for carried in block.carried_sequence_aliases
                    if retained_values is None
                    or all(
                        int(value_id) in retained_values
                        for value_id in carried
                    )
                ),
            )
        if isinstance(block, LoopBlock):
            body = project(block.body)
            has_structural_body = any(
                str(binding[2]) == str(block.induction)
                and str(binding[3]) == "induction"
                for binding in program.projected_iterable_bindings
            )
            if (
                body is None
                and not block.sequence_mutations
                and not has_structural_body
            ):
                return None
            return LoopBlock(
                block.induction,
                block.start,
                block.stop,
                block.step,
                body or SequenceBlock(()),
                carried_aliases=tuple(
                    (updated, initial)
                    for updated, initial in block.carried_aliases
                    # The UPDATE is what the body must produce; requiring the
                    # initial too dropped every reduction whose seed folded to
                    # a constant (max/sum seeds are literal zeros).  A carried
                    # value whose only consumer is its LoopResult port is not
                    # in retained_values at all, yet the port IS its
                    # retention: the loop itself declares the continuation.
                    if retained_values is None
                    or int(updated) in retained_values
                    or any(
                        int(updated) == int(port_updated)
                        for _port, _init, port_updated in block.result_ports
                    )
                ),
                carried_seeds=block.carried_seeds,
                result_ports=block.result_ports,
                parallel_iterations=block.parallel_iterations,
                dispatch_shell=block.dispatch_shell,
                recursion_region_id=block.recursion_region_id,
                schedule_preference=block.schedule_preference,
                sequence_mutations=block.sequence_mutations,
                comparison=block.comparison,
                terminal_controls=tuple(
                    projected
                    for terminal in block.terminal_controls
                    if (projected := project(terminal)) is not None
                ),
                source_loop_node_id=block.source_loop_node_id,
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
                carried_aliases=tuple(
                    (updated, initial)
                    for updated, initial in block.carried_aliases
                    # The UPDATE is what the body must produce; requiring the
                    # initial too dropped every reduction whose seed folded to
                    # a constant (max/sum seeds are literal zeros).  A carried
                    # value whose only consumer is its LoopResult port is not
                    # in retained_values at all, yet the port IS its
                    # retention: the loop itself declares the continuation.
                    if retained_values is None
                    or int(updated) in retained_values
                    or any(
                        int(updated) == int(port_updated)
                        for _port, _init, port_updated in block.result_ports
                    )
                ),
                carried_seeds=block.carried_seeds,
                result_ports=block.result_ports,
                recursion_region_id=block.recursion_region_id,
                predicate_expression=block.predicate_expression,
                sequence_mutations=block.sequence_mutations,
                source_loop_node_id=block.source_loop_node_id,
                terminal_controls=tuple(
                    projected
                    for terminal in block.terminal_controls
                    if (projected := project(terminal)) is not None
                ),
            )
        if isinstance(block, LoopControlBlock):
            if (
                block.predicate_value_id is not None
                and block.predicate_expression is None
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
        if isinstance(block, SequenceMutationBlock):
            return block
        if isinstance(block, SequenceQueryBlock):
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
        tuple(
            binding
            for binding in program.projected_iterable_bindings
            if str(binding[2]) in active_inductions
        ),
        program.specialized_conditional_node_ids,
    )


def overlay_scheduled_control(
    region_indices: Iterable[int],
    controls: Iterable[ControlProgram],
    *,
    known_nesting: "Mapping[int, Iterable[int]] | None" = None,
) -> ControlProgram:
    """Overlay planned control blocks on the flat scheduled region order.

    ``known_nesting``, if given, maps a control's index (into ``controls``)
    to the indices of controls known -- by real structure, not inferred
    here -- to be lexically nested directly inside it. Region-set strict
    containment (``child < parent``) is the ordinary signal for "this
    control is nested inside that one", but it is only a proxy: a loop
    whose entire body is another loop (``while a: while b: ...`` with
    nothing of its own between them) computes the *same* region set as
    its child, not a superset, since it contributes no region of its own.
    Strict-subset containment cannot tell those two controls apart -- ``<``
    is ``False`` in both directions for equal sets -- so without this hint
    they are wrongly treated as independent siblings both claiming the
    same regions, which is exactly the "maximal control blocks overlap
    without containment" failure this parameter exists to prevent.
    """

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
    projected_iterable_bindings = []
    recursion_regions = []
    deployment_regions = []
    specialized_conditional_node_ids = []
    controls = tuple(controls)
    controlled_sets = tuple(
        frozenset(control.region_indices) for control in controls
    )
    direct_children = {
        int(parent): frozenset(int(child) for child in children)
        for parent, children in (known_nesting or {}).items()
    }
    nested_children_overall = frozenset(
        child
        for children in direct_children.values()
        for child in children
    )

    def _nested_in(child: int, parent: int) -> bool:
        if child == parent:
            return False
        if controlled_sets[child] < controlled_sets[parent]:
            return True
        return child in direct_children.get(parent, ())

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
                # A leaf marker returns ``None`` after consuming the nested
                # region span, so this sequence owns insertion at that exact
                # lexical position.  A composite child returns its rewritten
                # block and ``consumed=True`` because it already inserted the
                # nested root internally; inserting again here duplicates the
                # complete subtree beside itself.
                if consumed and projected is None and not inserted:
                    children.append(nested_root)
                    inserted = True
                if projected is not None:
                    children.append(projected)
                    inserted |= consumed
            return SequenceBlock(tuple(children)), inserted
        if isinstance(block, ConditionalBlock):
            body, body_consumed = embed(
                block.body, nested_root, nested_regions
            )
            orelse = None
            else_consumed = False
            if block.orelse is not None:
                orelse, else_consumed = embed(
                    block.orelse, nested_root, nested_regions
                )
            return (
                ConditionalBlock(
                    block.predicate_value_id,
                    body or SequenceBlock(()),
                    orelse,
                    block.expect_true,
                    block.predicate_expression,
                    block.carried_aliases,
                    block.source_node_id,
                    block.carried_sequence_aliases,
                ),
                body_consumed or else_consumed,
            )
        if isinstance(block, LoopBlock):
            body, consumed = embed(
                block.body, nested_root, nested_regions
            )
            if body is None:
                body = nested_root if consumed else SequenceBlock(())
            return (
                LoopBlock(
                    block.induction,
                    block.start,
                    block.stop,
                    block.step,
                    body,
                    carried_aliases=block.carried_aliases,
                    result_ports=block.result_ports,
                    carried_seeds=block.carried_seeds,
                    parallel_iterations=block.parallel_iterations,
                    dispatch_shell=block.dispatch_shell,
                    recursion_region_id=block.recursion_region_id,
                    schedule_preference=block.schedule_preference,
                    sequence_mutations=block.sequence_mutations,
                    comparison=block.comparison,
                    terminal_controls=block.terminal_controls,
                    source_loop_node_id=block.source_loop_node_id,
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
            if condition is None:
                condition = (
                    nested_root if condition_consumed else SequenceBlock(())
                )
            if body is None:
                body = nested_root if body_consumed else SequenceBlock(())
            return (
                WhileBlock(
                    block.predicate_value_id,
                    condition,
                    body,
                    carried_aliases=block.carried_aliases,
                    result_ports=block.result_ports,
                    carried_seeds=block.carried_seeds,
                    recursion_region_id=block.recursion_region_id,
                    predicate_expression=block.predicate_expression,
                    sequence_mutations=block.sequence_mutations,
                    source_loop_node_id=block.source_loop_node_id,
                    terminal_controls=block.terminal_controls,
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
                if projected is None:
                    projected = nested_root if consumed else SequenceBlock(())
                cases.append((value, projected))
                consumed_any |= consumed
            default = None
            if block.default is not None:
                default, consumed = embed(
                    block.default, nested_root, nested_regions
                )
                if default is None:
                    default = nested_root if consumed else SequenceBlock(())
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
                if projected is None:
                    projected = nested_root if consumed else SequenceBlock(())
                lanes.append(projected)
                consumed_any |= consumed
            return (
                ParallelDeployment(tuple(lanes), block.schedule_preference),
                consumed_any,
            )
        if isinstance(block, CallBlock):
            callee, consumed = embed(
                block.callee, nested_root, nested_regions
            )
            if callee is None:
                callee = nested_root if consumed else SequenceBlock(())
            return (
                CallBlock(
                    block.callsite_id,
                    callee,
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
            and _nested_in(child, index)
            and not any(
                _nested_in(child, middle) and _nested_in(middle, index)
                for middle in range(len(controlled_sets))
                if middle not in {index, child}
            )
        ]
        for child in sorted(
            candidates,
            key=lambda item: min(
                positions[region] for region in controlled_sets[item]
            ),
        ):
            child_regions = controlled_sets[child]
            scope_paths = _marker_scope_paths(root, child_regions)
            distinct_scopes = sorted(set(scope_paths.values()))
            if len(distinct_scopes) > 1:
                by_scope: dict[tuple[str, ...], list[int]] = {}
                for region, scope in scope_paths.items():
                    by_scope.setdefault(scope, []).append(region)
                raise ValueError(
                    "nested control's regions span "
                    f"{len(distinct_scopes)} sequence scopes of the parent "
                    f"(control index {child}, regions "
                    f"{tuple(sorted(child_regions))}): "
                    + "; ".join(
                        f"{' > '.join(scope)}: regions {sorted(regions)}"
                        for scope, regions in sorted(by_scope.items())
                    )
                    + ". The schedule and the conditional compartments "
                    "disagree about where these regions live; embed "
                    "cannot insert one planner-owned control body into "
                    "more than one scope."
                )
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
        if regions
        and index not in nested_children_overall
        and not any(
            _nested_in(index, other)
            for other in range(len(controlled_sets))
            if other != index
        )
    ]
    for index, control in enumerate(controls):
        controlled = tuple(control.region_indices)
        uniforms.extend(control.uniforms)
        aliases.extend(control.value_aliases)
        iterable_bindings.extend(control.iterable_bindings)
        static_iterable_bindings.extend(control.static_iterable_bindings)
        collection_bindings.extend(control.collection_bindings)
        closure_iterable_bindings.extend(
            control.closure_iterable_bindings
        )
        projected_iterable_bindings.extend(
            control.projected_iterable_bindings
        )
        recursion_regions.extend(control.recursion_regions)
        deployment_base = len(deployment_regions)
        deployment_regions.extend(
            replace(region, region_id=deployment_base + offset)
            for offset, region in enumerate(control.deployment_regions)
        )
        specialized_conditional_node_ids.extend(
            control.specialized_conditional_node_ids
        )
        if not controlled:
            continue
        missing = set(controlled) - set(order)
        if missing:
            raise ValueError(
                "control overlay does not partition the schedule: "
                f"missing={sorted(missing)!r}"
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
    # Controls with no numerical region can still own complete compiled work:
    # resident sequence mutation, or an empty loop body into which hierarchy
    # composition will insert a source-linked CallBlock.  They have no schedule
    # marker to replace, so retain their roots explicitly.  A genuinely empty
    # loop remains filtered unless its induction appears in a projected
    # iterable binding, which is the loop composer's structural proof that a
    # retained body construct still owns this iteration.
    call_only_inductions = {
        str(induction)
        for control in controls
        for _iterable, _target, induction, projection
        in control.projected_iterable_bindings
        if str(projection) == "induction"
    }
    blocks = [
        control.root
        for control in controls
        if not control.region_indices
        and (
            isinstance(control.root, LoopBlock)
            and (
                bool(control.root.sequence_mutations)
                or str(control.root.induction) in call_only_inductions
            )
            or isinstance(control.root, WhileBlock)
            and bool(control.root.sequence_mutations)
        )
    ]
    for region_index in order:
        replacement = replacements.get(region_index)
        if replacement is not None:
            blocks.append(replacement)
        elif region_index not in covered:
            blocks.append(StatementBlock((
                f"__scheduled_region_{region_index}__",
            )))

    def stable_key(value):
        if isinstance(value, slice):
            return ("slice", value.start, value.stop, value.step)
        if isinstance(value, tuple):
            return ("tuple", tuple(stable_key(item) for item in value))
        if isinstance(value, list):
            return ("list", tuple(stable_key(item) for item in value))
        if isinstance(value, dict):
            return (
                "dict",
                tuple(sorted(
                    (stable_key(key), stable_key(item))
                    for key, item in value.items()
                )),
            )
        try:
            hash(value)
        except TypeError:
            return (type(value).__qualname__, repr(value))
        return ("value", value)

    def unique_unhashable(values):
        seen = set()
        unique = []
        for value in values:
            key = stable_key(value)
            if key not in seen:
                seen.add(key)
                unique.append(value)
        return tuple(unique)

    return ControlProgram(
        root=SequenceBlock(tuple(blocks)),
        region_indices=order,
        uniforms=tuple(dict.fromkeys(uniforms)),
        value_aliases=tuple(dict.fromkeys(aliases)),
        iterable_bindings=tuple(dict.fromkeys(iterable_bindings)),
        static_iterable_bindings=unique_unhashable(static_iterable_bindings),
        collection_bindings=tuple(dict.fromkeys(collection_bindings)),
        closure_iterable_bindings=tuple(
            dict.fromkeys(closure_iterable_bindings)
        ),
        recursion_regions=tuple(dict.fromkeys(recursion_regions)),
        deployment_regions=tuple(deployment_regions),
        projected_iterable_bindings=tuple(
            dict.fromkeys(projected_iterable_bindings)
        ),
        specialized_conditional_node_ids=tuple(dict.fromkeys(
            int(node_id) for node_id in specialized_conditional_node_ids
        )),
    )


def place_validations_after_region_producers(
    program: ControlProgram,
    validations: Iterable[ValidationBlock],
    *,
    predicate_regions: Mapping[int, int],
) -> ControlProgram:
    """Place each guard after the region that computes its predicate.

    Validation is lexical compiled control, not an entrypoint precondition.
    In particular, a predicate can itself be the output of an earlier
    numerical region.  Prefixing every guard to ``program.root`` reads that
    result before it exists and quietly turns the canonical SSA identity into
    an invented public argument.  Correlate by the existing value identity
    and splice the guard directly after the unique scheduled region marker.

    A predicate with no numerical producer is already an authored input (or a
    control expression over authored inputs), so its guard remains an entry
    prelude.  A predicate reported in two regions is not a schedule at all and
    is refused rather than guessed.
    """

    pending_by_region: dict[int, list[ValidationBlock]] = {}
    prelude: list[ValidationBlock] = []
    for validation in validations:
        region = predicate_regions.get(int(validation.predicate_value_id))
        if region is None:
            prelude.append(validation)
        else:
            pending_by_region.setdefault(int(region), []).append(validation)

    placed: set[int] = set()

    def marker(line: str) -> int | None:
        prefix = "__scheduled_region_"
        suffix = "__"
        if not (line.startswith(prefix) and line.endswith(suffix)):
            return None
        try:
            return int(line[len(prefix):-len(suffix)])
        except ValueError:
            return None

    def visit(block: ControlBlock) -> ControlBlock:
        if isinstance(block, StatementBlock):
            expanded: list[ControlBlock] = []
            residual: list[str] = []
            for line in block.lines:
                residual.append(line)
                region = marker(line)
                if region is None or region not in pending_by_region:
                    continue
                expanded.append(StatementBlock(tuple(residual)))
                residual = []
                if region in placed:
                    raise ValueError(
                        "scheduled region marker occurs more than once while "
                        f"placing validation predicates: region={region}"
                    )
                expanded.extend(pending_by_region[region])
                placed.add(region)
            if residual:
                expanded.append(StatementBlock(tuple(residual)))
            if len(expanded) == 1:
                return expanded[0]
            return SequenceBlock(tuple(expanded))
        if isinstance(block, SequenceBlock):
            return replace(block, blocks=tuple(map(visit, block.blocks)))
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=visit(block.body),
                orelse=(visit(block.orelse) if block.orelse is not None else None),
            )
        if isinstance(block, LoopBlock):
            return replace(block, body=visit(block.body))
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=visit(block.condition),
                body=visit(block.body),
            )
        if isinstance(block, StateMachineTick):
            return replace(
                block,
                cases=tuple((value, visit(body)) for value, body in block.cases),
                default=(visit(block.default) if block.default is not None else None),
            )
        if isinstance(block, ParallelDeployment):
            return replace(block, lanes=tuple(map(visit, block.lanes)))
        if isinstance(block, CallBlock):
            return replace(block, callee=visit(block.callee))
        return block

    root = visit(program.root)
    missing = set(pending_by_region) - placed
    if missing:
        raise ValueError(
            "validation predicate producer regions are absent from the "
            f"scheduled control tree: regions={sorted(missing)!r}"
        )
    if prelude:
        root = SequenceBlock((*prelude, root))
    return replace(program, root=root)


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
    if "__turing_external_call__" not in scope:
        from .shell_external_references import PythonShellExternalReferenceResolver

        external_resolver = PythonShellExternalReferenceResolver()
        scope["__turing_external_call__"] = external_resolver.call
    else:
        external_resolver = None
    if abstract_tensor_backend is not None:
        scope["AbstractTensor"] = AbstractTensor
    exec(compile(source, f"<compiled-shell:{function_name}>", "exec"), scope)
    result = scope[function_name]
    result.__compiled_shell_source__ = source
    result.__external_reference_resolver__ = external_resolver
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
    "ControlSequenceMutation",
    "CallBlock",
    "ExternalReferenceCallBlock",
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
    "SequenceMutationBlock",
    "SequenceQueryBlock",
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
