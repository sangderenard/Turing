"""Target-neutral compiled-shell control structure and source rendering.

The planner owns control flow.  Backends render this structure; they must not
rediscover loops, reorder scheduled regions, or substitute a host-language
coordinator after planning has selected a compiled target.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from .deployment_frame import DeploymentFrame, DeploymentJoin
from .live_tiles import TiledObject


class ControlOverlayScopeError(ValueError):
    """One control body has region markers in several insertion scopes."""

    def __init__(
        self,
        message: str,
        *,
        control_index: int,
        region_indices: Iterable[int],
        scopes: Mapping[tuple[str, ...], Iterable[int]],
    ) -> None:
        super().__init__(message)
        self.control_index = int(control_index)
        self.region_indices = tuple(sorted(map(int, region_indices)))
        self.scopes = {
            tuple(map(str, scope)): tuple(sorted(map(int, regions)))
            for scope, regions in scopes.items()
        }


@dataclass(frozen=True)
class AuthoredControlSite:
    """Exact authored execution slot for one control-to-control edge.

    ``known_nesting`` says which control is structurally related to which;
    this record preserves the part of that source fact which cannot be
    reconstructed from region containment.  In particular, a comprehension
    used as another comprehension's iterable is a prelude, while one in its
    filter or element executes inside each owning iteration.
    """

    parent_source_node_id: int
    child_source_node_id: int
    slot: str
    ordinal: int | None = None
    path: tuple[str | int, ...] = ()


class ControlTarget(str, Enum):
    PYTHON = "python"
    C = "c"
    GLSL = "glsl"
    # Fortran renders the same control structure with `do`/`select case`.  It
    # is worth having as a launch environment because its arrays cannot alias,
    # which is the freedom the C and LLVM routes must assert explicitly.
    FORTRAN = "fortran"


@dataclass(frozen=True)
class StatementBlock(TiledObject):
    """Already-lowered statements in planner-approved execution order."""

    lines: tuple[str, ...]


@dataclass(frozen=True)
class SequenceBlock(TiledObject):
    blocks: tuple["ControlBlock", ...]


@dataclass(frozen=True)
class ControlExpression(TiledObject):
    """Typed scalar expression evaluated by the compiled control backend."""

    op: str
    operands: tuple["ControlExpression", ...] = ()
    value_id: int | None = None
    literal: bool | int | float | tuple | list | None = None
    # Where a ``value`` leaf was read: ``(consumer node, operand role,
    # ordinal)``, the key of its ``lexical_read_binding`` row.  Structure
    # only -- the binding it read is the book's fact, looked up by the
    # lowering.  Not part of equality: two leaves reading one value compute
    # the same expression wherever they sit.
    read: tuple | None = field(default=None, compare=False)
    # Exact ProgramABI ownership for an optional-presence leaf.  A Boolean
    # control value can outlive the graph node that introduced it, so carry
    # the semantic owner through Control IR instead of re-deriving it from an
    # integer id during SSA construction.
    program_abi_parameter: str | None = None
    program_abi_field: str | None = None
    program_abi_record: str | None = None
    optional_payload_value_id: int | None = None


@dataclass(frozen=True)
class ConditionalBlock(TiledObject):
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
    # Source call nodes lexically inside each arm. A planned call (a
    # ``__plan_callsite_N__`` statement) is otherwise scheduled relative to
    # numerical regions and loops only; an arm whose entire content is a
    # call (``if d is not None: return d(...)``) has no region for the
    # scheduler to anchor on, so the callsite scheduler places these inside
    # the arm that owns them.
    body_callsite_ids: tuple[int, ...] = ()
    orelse_callsite_ids: tuple[int, ...] = ()
    # (true-arm value, false-arm value, expression result). Unlike a carried
    # assignment, an expression has no pre-branch value to snapshot or rebind.
    result_aliases: tuple[tuple[int, int, int], ...] = ()
    # Merged ids of ``result_aliases`` that are lists merged from branches:
    # the merge is a handle onto the taken arm's own storage, each descriptor
    # member one Phi of the arms' members (not a copy of the arm).
    sequence_handle_results: tuple[int, ...] = ()
    # Scheduled regions that produce the predicate value: prerequisites of
    # this block wherever the flat schedule places them.  A conditional's
    # membership is its arm regions; without this, ordering placed the
    # ``if`` before the region computing its own predicate's input
    # (``exchange_time_bound``: ``binding.any()`` scheduled after the ``if``
    # whose test reads it).
    predicate_region_indices: tuple[int, ...] = ()
    # Aligned with ``carried_aliases``: for a field-carried merge, the
    # ``reducer_field_state`` cells (``Ref``) the reducer's merge named --
    # (true-arm cell, false-arm cell, merged cell); None for a name-carried
    # alias or when the reducer posted no cells.  Lowering reads each arm's
    # SSA version at its cell instead of looking a graph id up by hand.
    # Last field on purpose: positional constructions above never reach it.
    carried_field_cells: tuple[Any, ...] = ()
    # Exact structural producers for name-carried arm versions which are not
    # owned by a numerical region or a scheduled call. Each pair is
    # ``(graph value id, expression)`` and is evaluated inside the named arm
    # before its Phi operand is selected. The reducer's Phi parent edge is
    # the authority for membership; this is not rebuilt from the binding
    # name. String/record producers stay absent until their native sequence/
    # record representation is admitted -- they must not be forced through
    # the scalar expression vocabulary.
    body_value_expressions: tuple[tuple[int, ControlExpression], ...] = ()
    orelse_value_expressions: tuple[tuple[int, ControlExpression], ...] = ()
    # Exact SOURCE_OPTIONAL_GLYPH_CARRY cells removed from the scalar-carried
    # lane.  Their payload is resident sequence storage and their value is an
    # explicit presence bit; lowering must not manufacture a scalar Phi for
    # the sequence identity.
    optional_glyph_carry_cells: tuple[Any, ...] = ()
    # ``"logical_and"`` / ``"logical_or"`` when this block IS a short-circuit
    # operator (``_short_circuit_control_program``): its result merge is the
    # operator's definition and carries that semantic family.
    short_circuit_operator: str | None = None


@dataclass(frozen=True)
class ControlSequenceMutation(TiledObject):
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
    # Accepted object representation on the shared book, not a second schema.
    object_contract_cell: Any = None
    # Exact source recipe cell for compiler-authored resident sequence
    # composition (currently dynamic glyph sequences).  This is distinct
    # from ``object_contract_cell``: no object is admitted, and lowering may
    # only consume the sequence operations named by this source cell.
    source_contract_cell: Any = None
    vocabulary_contract_cell: Any = None
    # The ``glyph_sequence_step`` cell of this one step of a compiler-authored
    # glyph composition, derived from ``source_contract_cell``.  Like a
    # scalar write's ``field_state_cell`` it owns the step's control block,
    # so a ``clear`` and its ``extend`` steps (one effect node) are rows of
    # their own rather than revisions of one.
    step_cell: Any = None
    # Native ``list.sort`` options retained from the authored call.  Mapping
    # ``__getitem__`` keys use ``argument_value_ids`` plus
    # ``argument_kind == "sort_key_mapping_getitem"``; a direct row
    # projection such as ``lambda row: row[1]`` records its resident column.
    sort_key_column: int | None = None
    sort_reverse: bool = False
    # ``(display node, ordinal)`` for one member append of an authored list
    # display, read from its SOURCE_AGGREGATE_LAYOUT birth.  The append is an
    # effect of the display: it executes at the display, in member order.
    # ``effect_node_id`` names the member's producer, which is neither unique
    # (``[k, 9]`` and ``[4, k]`` share ``k``; ``[n, 3, n]`` repeats ``n``)
    # nor positioned (a parameter has no authored coordinate).
    display_member: tuple[int, int] | None = None


@dataclass(frozen=True)
class LoopBlock(TiledObject):
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
    carried_seeds: tuple[tuple[int, bool | int | float], ...] = ()
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
    # Graph nodes of every source break/continue in this loop's body.  The
    # lowering checks each one was emitted (in an arm or at its lexical
    # position); a site nothing placed is a loud shortfall, never a loop
    # that silently runs to completion.
    control_site_ids: tuple[int, ...] = ()

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
class WhileBlock(TiledObject):
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
    carried_seeds: tuple[tuple[int, bool | int | float], ...] = ()
    recursion_region_id: int | None = None
    predicate_expression: ControlExpression | None = None
    sequence_mutations: tuple[ControlSequenceMutation, ...] = ()
    # Exact source ProcessGraph loop identity.  A source-linked call retains
    # this same identity in ``PlanCall.enclosing_loop_ids``; carrying it into
    # Control IR lets the SSA linker place that call in the authored while
    # body instead of guessing from a later result consumer.
    source_loop_node_id: int | None = None
    terminal_controls: tuple["LoopControlBlock", ...] = ()
    # Graph nodes of every source break/continue in this loop's body.  The
    # lowering checks each one was emitted (in an arm or at its lexical
    # position); a site nothing placed is a loud shortfall, never a loop
    # that silently runs to completion.
    control_site_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class LoopControlBlock(TiledObject):
    """A planner-owned ``break`` or ``continue`` edge.

    When ``predicate_value_id`` is absent the edge is unconditional.  A
    conditional edge branches only when the resident predicate is true.
    """

    action: str
    predicate_value_id: int | None = None
    expect_true: bool = True
    predicate_expression: ControlExpression | None = None
    source_action: str | None = None
    # ``action == "return"``: a source ``return`` inside a loop or
    # conditional. Lowering captures these slot values (function-output
    # order) on the edge and branches to the function's exit block, where
    # every return edge is merged per slot (control-aware result merging;
    # docs/PLAN_CONTROL_AWARE_RESULT_MERGING.md). A ``None`` slot is one the
    # reducer could not resolve at that return.
    return_value_ids: tuple[int | None, ...] = ()
    # ``break``/``continue``: the graph node of the source statement, and
    # the bindings live at that site as ((pre-loop identity, loop-carried
    # update), value-at-site) pairs -- the values the exit edge carries for
    # every name the arm rebound (reducer ``loop_break_sites``).  The
    # binding is the carried pair, not the pre-loop identity alone: two
    # names seeded from one value are distinct bindings.  A name that is
    # not carried is keyed (initial, initial).  A missing pair means the
    # site did not rebind that name.
    site_node_id: int | None = None
    site_values: tuple[tuple[tuple[int, int], int], ...] = ()
    # ``return``: the return SITE's identity cell -- the cell the reducer
    # keys the ``return_site_slot`` / ``return_site_field_state`` rows by
    # (the return construct, never the value it returns).  Lowering stamps
    # it on the return edge beside ``return_source_value_ids`` so a merge
    # predecessor names its own site.  None when no site row exists.
    return_site_cell: Any = None

    def __post_init__(self) -> None:
        if self.action not in {"break", "continue", "return"}:
            raise ValueError(
                "loop control action must be break, continue or return"
            )
        if self.source_action not in {
            None, "break", "continue", "loop-return", "return",
        }:
            raise ValueError("unknown source loop-control action")


@dataclass(frozen=True)
class StateMachineTick(TiledObject):
    """One compiled state transition, not a host polling loop.

    ``state`` is the spelling the renderers print (a loop induction name, a
    uniform).  ``state_value_id`` is the state itself: the graph value the
    selector reads, resolved through the book like a conditional's
    predicate.  ``source_node_id`` is the dispatch construct's graph node
    (the ``match`` statement), the tick's owner cell the way a
    conditional's ``source_node_id`` is its.  A tick carrying neither is a
    hand-built one (a coordinator's method switch): it has no owner to name
    and posts ``Unsourced(CONTROL_OWNER_UNKNOWN)`` like any ownerless block.
    """

    state: str
    cases: tuple[tuple[str, "ControlBlock"], ...]
    default: "ControlBlock | None" = None
    state_value_id: int | None = None
    source_node_id: int | None = None
    # Aligned with ``cases``: the graph node of each case literal (the
    # ``MatchValue`` constant), whose identity cell the case's blocks and the
    # equality predicate derive from.  None where a case has no node.
    case_value_ids: tuple[int | None, ...] = ()
    # One entry per scalar the arms rebind: ``(per-case arm value ids, the
    # default arm's value id, initial value id, merged value id)``.  An arm
    # that does not rebind the name repeats the initial id (the entered
    # version), exactly as ``ConditionalBlock.carried_aliases`` encodes a
    # missing arm; with no ``default`` the default slot is the initial id --
    # the version a state matching no case leaves in place.  Lowering joins
    # every arm that reaches the merge in one Phi.
    carried_aliases: tuple[
        tuple[tuple[int, ...], int, int, int], ...
    ] = ()


@dataclass(frozen=True)
class ParallelDeployment(TiledObject):
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
class ControlDeploymentLane(TiledObject):
    """One proven-independent lane in a backend-neutral deployment region."""

    index: int
    region_indices: tuple[int, ...] = ()
    value_ids: tuple[int, ...] = ()
    source_node_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ControlDeploymentRegion(TiledObject):
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
class CallBlock(TiledObject):
    """Planner-owned nested closure invocation with explicit value bindings."""

    callsite_id: int
    callee: "ControlBlock"
    argument_bindings: tuple[tuple[int, int], ...] = ()
    result_bindings: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ResourceScopeBlock(TiledObject):
    """Lexical resource cleanup, including nonlocal return/break/continue.

    Cleanup operations consume captured handles and have no published result.
    This represents generated context-manager cleanup, not Python exception
    handlers or a general finally body which can override a pending exit.
    """

    body: "ControlBlock"
    cleanup: tuple["DispatchBlock", ...]
    source_scope_id: int

    def __post_init__(self):
        if any(not isinstance(operation, DispatchBlock)
               or operation.result_value_id is not None for operation in self.cleanup):
            raise ValueError("resource cleanup requires result-free dispatcher operations")


@dataclass(frozen=True)
class DispatchBlock(TiledObject):
    """An ordered dispatcher operation with explicit SSA handle/value ports.

    The operation is semantic (for example condition_wait), not a C symbol.
    Backend admission must resolve its synchronization and lifetime contract.
    This block grants no permission to serialize communicating tasks.
    """

    callsite_id: int
    operation: str
    argument_value_ids: tuple[int, ...] = ()
    keyword_argument_value_ids: tuple[tuple[str, int], ...] = ()
    result_value_id: int | None = None
    result_dtype: str = "opaque_ref"


@dataclass(frozen=True)
class ExternalReferenceCallBlock(TiledObject):
    """One authored call through the shell external-reference capability."""

    callsite_id: int
    identity: str
    argument_value_ids: tuple[int, ...]
    keyword_argument_value_ids: tuple[tuple[str, int], ...]
    result_value_id: int
    result_dtype: str
    argument_sequence_ids: tuple[tuple[int, int, str], ...] = ()
    shell_abi: str = "turing-shell-io-abi.external_references"
    external_domain: str = "host_system"
    native_abi: str = ""
    runtime_owner: str = ""
    shell_profiles: tuple[str, ...] = ()
    # Exact native failure classes declared by the extraction receipt.  The
    # integer is the native status tag and the string is the declaration
    # identity (for example ``builtins.OSError``).  Status is a compiler-owned
    # result distinct from the authored call's payload.
    failure_statuses: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class NativeExceptionHandler(TiledObject):
    """One exact authored ``except`` arm of a retained native Try."""

    body: "ControlBlock"
    source_handler_node_id: int
    # The source graph identities of the exception class declarations.  A
    # bare except has no declarations and therefore an empty tuple.
    exception_type_value_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class NativeTryBlock(TiledObject):
    """Catchable native status control owned by one authored ``ast.Try``.

    Body, handlers, orelse and finalbody remain separate lexical arms.  A
    backend must run orelse only after a successful body, dispatch a failed
    native status to the first matching handler, and traverse finalbody for
    every normal or handled path.  An unmatched status propagates after the
    finalbody; it must never be converted to success or a process abort.
    """

    body: "ControlBlock"
    handlers: tuple[NativeExceptionHandler, ...]
    orelse: "ControlBlock | None"
    finalbody: "ControlBlock"
    source_try_node_id: int


@dataclass(frozen=True)
class ValidationBlock(TiledObject):
    """Device-side predicate whose failure is reported through shell errors."""

    predicate_value_id: int
    error_code: int
    expect_true: bool = True
    predicate_expression: ControlExpression | None = None
    extraction_identity: str | None = None
    # Exact source values consumed only by the authored exception diagnostic.
    # Native validation reports ``error_code`` and does not materialize them.
    diagnostic_value_ids: tuple[int, ...] = ()


def validation_diagnostic_value_ids(
    statement: ast.Raise,
    node_for_expression: Callable[[ast.AST], int | None],
) -> tuple[int, ...]:
    """Resolve the source values used only to author a validation message.

    Validation discovery has two planner lanes: ordinary guards and guards
    lexically owned by retained loops.  Both must publish the same identities
    to ``control_block``; otherwise resident planning sees different use
    contracts for the same authored raise depending on its nesting.
    """

    if not isinstance(statement, ast.Raise):
        return ()
    raised = statement.exc
    expressions = (
        (*raised.args, *(keyword.value for keyword in raised.keywords))
        if isinstance(raised, ast.Call)
        else (raised,) if isinstance(raised, ast.expr)
        else ()
    )
    return tuple(dict.fromkeys(
        int(value_id)
        for expression in expressions
        for value_id in (node_for_expression(expression),)
        if value_id is not None
    ))


@dataclass(frozen=True)
class SequenceMutationBlock(TiledObject):
    """One lexical resident-sequence effect outside implicit loop storage."""

    mutation: ControlSequenceMutation


@dataclass(frozen=True)
class ScalarFieldWriteBlock(TiledObject):
    """Publish a scalar field version at its authored effect site.

    ``field_value_id`` names resident parameter storage when the receiver has a
    physical in/out ABI.  A local or returned record has no such cell in this
    control function; ``None`` still publishes the exact field-state version
    for conditional joins and later record-return materialization.
    """

    field_value_id: int | None
    value_expression: ControlExpression
    dtype: str
    effect_node_id: int
    # The ``reducer_field_state`` cell (a ``Ref``) this write IS: the WRITTEN
    # revision the reducer posted for the SetAttr.  Control SSA publishes the
    # emitted value under this cell on ``ssa_field_version``; None when the
    # reducer has not posted one for the effect node.
    field_state_cell: Any = None


@dataclass(frozen=True)
class SequenceQueryBlock(TiledObject):
    """Read one scalar fact from a resident sequence at lexical position."""

    result_value_id: int
    sequence_value_id: int
    operation: str
    default_value_id: int | None = None
    source_call_node_id: int | None = None
    extraction_identity: str | None = None
    result_alias_ids: tuple[int, ...] = ()
    # Exact reducer-owned member identities when the query returns one
    # aggregate row. The aggregate root has no independent physical SSA
    # value; resident columns publish under these members instead.
    result_member_ids: tuple[int, ...] = ()
    producer_loop_node_id: int | None = None
    # Exact authored key identities for a resident keyed-table read.  These
    # are defined at this lexical position (often by a loop projection), so a
    # backend must not hoist the lookup to function entry.
    key_value_ids: tuple[int, ...] = ()
    # Membership is the same resident-table query as lookup, but publishes a
    # boolean and may invert it for ``not in``.  It must remain a lexical
    # block because its key can be a loop projection.
    negate: bool = False
    # A first-or-default query over a derived record sequence yields a tagged
    # integer row handle.  The default arm uses -1 and subsequent source
    # ``is [not] None`` tests consume that tag.
    row_handle: bool = False
    # Ordered scalar operands surrounding a starred resident sequence. For
    # ``max(a, *items, b)`` these are ``(a,)`` and ``(b,)``. Keeping the two
    # sides distinct preserves Python's left-to-right comparison semantics,
    # including equal values and NaNs retaining the incumbent.
    reduction_prefix_value_ids: tuple[int, ...] = ()
    reduction_suffix_value_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in {
            "length", "truth", "first_or_default", "lookup", "membership",
            "maximum", "any", "first", "sum", "count", "index",
        }:
            raise ValueError("unknown resident sequence query")
        if self.operation == "first_or_default" and self.default_value_id is None:
            raise ValueError("first_or_default requires an explicit default")
        if self.operation in {"lookup", "membership", "count", "index"} and not self.key_value_ids:
            raise ValueError(
                f"{self.operation} requires at least one exact key identity"
            )
        if self.operation == "maximum" and not self.reduction_prefix_value_ids:
            raise ValueError("maximum requires an incumbent before the starred sequence")


@dataclass(frozen=True)
class StreamPublishBlock(TiledObject):
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
    # An authored aggregate has no scalar value of its own.  Its reducer
    # layout is the payload: retain those exact member identities so SSA does
    # not manufacture a public formal for the tuple/list syntax node.
    payload_value_ids: tuple[int, ...] = ()


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
    | DispatchBlock
    | ResourceScopeBlock
    | ExternalReferenceCallBlock
    | NativeTryBlock
    | ValidationBlock
    | SequenceMutationBlock
    | SequenceQueryBlock
    | ScalarFieldWriteBlock
    | StreamPublishBlock
)


@dataclass(frozen=True)
class RecursionRegion(TiledObject):
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
class ControlProgram(TiledObject):
    __live_storage_root__ = True
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
    collection_bindings: tuple[
        tuple[int, int, str, int | str], ...
    ] = ()
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
    # A control that owns NO numerical region (a guard clause whose arm is
    # only a ``return``, or a call) has no marker to replace in the
    # schedule.  ``anchor_region`` names the first scheduled region that
    # lexically follows it; the overlay inserts the control immediately
    # before that region (or at the end of its scope when None) instead of
    # dropping it.  Declared last: ControlProgram is built positionally in
    # places, so a new field must never shift the existing ones.
    anchor_region: int | None = None
    # (counter target value id, start value id) for ``enumerate(a, start)``:
    # the counter target is the loop induction plus ``start``.  Declared
    # after ``anchor_region`` for the positional-build reason above.
    induction_offsets: tuple[tuple[int, int], ...] = ()


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
        control.induction_offsets,
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
        elif isinstance(block, ScalarFieldWriteBlock):
            if block.field_value_id is not None:
                values.add(int(block.field_value_id))
            values.add(int(block.effect_node_id))
            expression_values(block.value_expression)
        elif isinstance(block, SequenceQueryBlock):
            values.add(int(block.result_value_id))
            values.update(int(value_id) for value_id in block.result_alias_ids)
            values.update(int(value_id) for value_id in block.result_member_ids)
            values.add(int(block.sequence_value_id))
            values.update(int(value_id) for value_id in block.key_value_ids)
            values.update(
                int(value_id) for value_id
                in block.reduction_prefix_value_ids
            )
            values.update(
                int(value_id) for value_id
                in block.reduction_suffix_value_ids
            )
            if block.default_value_id is not None:
                values.add(int(block.default_value_id))
        elif isinstance(block, StreamPublishBlock):
            values.add(int(block.value_id))
            values.update(int(value_id) for value_id in block.payload_value_ids)
            if block.count_value_id is not None:
                values.add(int(block.count_value_id))
            if block.predicate_value_id is not None:
                values.add(int(block.predicate_value_id))
        elif isinstance(block, (ExternalReferenceCallBlock, DispatchBlock)):
            values.update(int(value_id) for value_id in block.argument_value_ids)
            values.update(
                int(value_id)
                for _name, value_id in block.keyword_argument_value_ids
            )
            if isinstance(block, DispatchBlock) and block.result_value_id is not None:
                values.add(int(block.result_value_id))
        elif isinstance(block, ResourceScopeBlock):
            visit(block.body)
            for operation in block.cleanup:
                visit(operation)
        elif isinstance(block, NativeTryBlock):
            visit(block.body)
            for handler in block.handlers:
                values.update(map(int, handler.exception_type_value_ids))
                visit(handler.body)
            if block.orelse is not None:
                visit(block.orelse)
            visit(block.finalbody)
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
            # The value each rebound name holds at the site is what the
            # break/continue edge carries: a consumer of planned control.
            # (``if dt > 0.2: dt = dt * 0.5; continue`` -- once no
            # conditional merge consumed ``dt * 0.5``, its region was
            # dropped and the continue edge carried the header value.)
            values.update(int(value) for _binding, value in block.site_values)
        elif isinstance(block, StateMachineTick):
            if block.state_value_id is not None:
                values.add(int(block.state_value_id))
            values.update(
                int(value_id) for value_id in block.case_value_ids
                if value_id is not None
            )
            for arm_ids, default_id, initial_id, merged_id in (
                block.carried_aliases
            ):
                values.update(int(value_id) for value_id in arm_ids)
                values.update((
                    int(default_id), int(initial_id), int(merged_id),
                ))
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
class ControlUniform(TiledObject):
    name: str
    value_id: int
    dtype: str = "int"


@dataclass(frozen=True)
class RegionCode(TiledObject):
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


# ---------------------------------------------------------------------------
# Step 9, part A (2) (plan 100, section 2): every control block is a row on
# ``control_block``, its position a ``control_block_placement`` revision, the
# program a ``control_program`` row.  The tree every builder and rewriter
# returns is the materialized view; ``post_control_program`` is the one
# writer, called once at each return.
# ---------------------------------------------------------------------------

#: The scope a program is tagged under when its graph has no read scope.
_UNSCOPED_CONTROL = ("unscoped_control", 0)
_CALLSITE_MARKER_PREFIX = "__plan_callsite_"


def _control_block_kind(block: Any) -> Any:
    """The ``ControlBlockKind`` of one block; None for a ``SequenceBlock``
    (a container, flattened for placement) or an unknown object."""

    from .concordance_declarations import ControlBlockKind as Kind

    return {
        StatementBlock: Kind.STATEMENT,
        ConditionalBlock: Kind.CONDITIONAL,
        LoopBlock: Kind.LOOP,
        WhileBlock: Kind.WHILE,
        LoopControlBlock: Kind.LOOP_CONTROL,
        StateMachineTick: Kind.STATE_MACHINE_TICK,
        ParallelDeployment: Kind.PARALLEL_DEPLOYMENT,
        CallBlock: Kind.CALL,
        DispatchBlock: Kind.DISPATCH,
        ResourceScopeBlock: Kind.RESOURCE_SCOPE,
        ExternalReferenceCallBlock: Kind.EXTERNAL_REFERENCE_CALL,
        NativeTryBlock: Kind.NATIVE_TRY,
        ValidationBlock: Kind.VALIDATION,
        SequenceMutationBlock: Kind.SEQUENCE_MUTATION,
        SequenceQueryBlock: Kind.SEQUENCE_QUERY,
        ScalarFieldWriteBlock: Kind.SCALAR_FIELD_WRITE,
        StreamPublishBlock: Kind.STREAM_PUBLISH,
    }.get(type(block))


def _flatten_control_sequence(block: Any) -> tuple:
    """The blocks of one arm with every nested ``SequenceBlock`` flattened:
    the ordinal a placement records (plan 100, 2.1)."""

    if block is None:
        return ()
    if isinstance(block, SequenceBlock):
        return tuple(
            item
            for child in block.blocks
            for item in _flatten_control_sequence(child)
        )
    return (block,)


def _control_block_arms(block: Any) -> tuple:
    """``(arm, index, flattened children)`` per arm of ``block``."""

    from .concordance_declarations import Arm

    if isinstance(block, ConditionalBlock):
        arms = [(Arm.BODY, 0, block.body)]
        if block.orelse is not None:
            arms.append((Arm.ORELSE, 0, block.orelse))
    elif isinstance(block, LoopBlock):
        arms = [
            (Arm.BODY, 0, block.body),
            (Arm.TERMINAL, 0, SequenceBlock(tuple(block.terminal_controls))),
        ]
    elif isinstance(block, WhileBlock):
        arms = [
            (Arm.CONDITION, 0, block.condition),
            (Arm.BODY, 0, block.body),
            (Arm.TERMINAL, 0, SequenceBlock(tuple(block.terminal_controls))),
        ]
    elif isinstance(block, StateMachineTick):
        arms = [
            (Arm.CASE, index, body)
            for index, (_state, body) in enumerate(block.cases)
        ]
        if block.default is not None:
            arms.append((Arm.DEFAULT, 0, block.default))
    elif isinstance(block, ParallelDeployment):
        arms = [(Arm.LANE, index, lane) for index, lane in enumerate(block.lanes)]
    elif isinstance(block, CallBlock):
        arms = [(Arm.CALLEE, 0, block.callee)]
    elif isinstance(block, ResourceScopeBlock):
        arms = [
            (Arm.BODY, 0, block.body),
            (Arm.CLEANUP, 0, SequenceBlock(tuple(block.cleanup))),
        ]
    elif isinstance(block, NativeTryBlock):
        arms = [(Arm.BODY, 0, block.body)]
        arms.extend(
            (Arm.HANDLER, index, handler.body)
            for index, handler in enumerate(block.handlers)
        )
        if block.orelse is not None:
            arms.append((Arm.ORELSE, 0, block.orelse))
        arms.append((Arm.FINALBODY, 0, block.finalbody))
    else:
        return ()
    return tuple(
        (arm, index, _flatten_control_sequence(body))
        for arm, index, body in arms
    )


def _callsite_marker(block: Any) -> int | None:
    if not isinstance(block, StatementBlock) or len(block.lines) != 1:
        return None
    line = block.lines[0]
    if not line.startswith(_CALLSITE_MARKER_PREFIX) or not line.endswith("__"):
        return None
    try:
        return int(line[len(_CALLSITE_MARKER_PREFIX):-2])
    except ValueError:
        return None


def control_callsite_markers(control: "ControlProgram | None") -> frozenset:
    """The planned callsites a ControlProgram already places: every
    ``__plan_callsite_N__`` marker in any block, arms of every construct
    (a state-machine tick's cases included) -- a marker present is placed,
    and the lexical scheduler must not place it a second time."""

    found: set = set()

    def walk(block: Any) -> None:
        for child in _flatten_control_sequence(block):
            marker = _callsite_marker(child)
            if marker is not None:
                found.add(int(marker))
            for _arm, _index, children in _control_block_arms(child):
                for grandchild in children:
                    walk(grandchild)

    if control is not None:
        walk(control.root)
    return frozenset(found)


def _describe_control_block(
    block: Any, *, cell: Any, region_cell: Any, callsite_cell: Any,
) -> Any:
    """``(kind, owner cell or None, reason, fact fields)`` of one block
    (plan 100, 2.2); ``owner`` None means the program cell stands in.

    ``cell(value id)``, ``region_cell(ordinal)`` and ``callsite_cell(id)``
    resolve ids to cells: ``post_control_program`` passes its graph-backed
    resolvers, ``control_block_cell`` read-only ones, so the one rule that
    names a block's owner keys both the writer's rows and the reader's
    lookups."""

    from .concordance_declarations import CONTROL_OWNER_UNKNOWN, REGION_CELL_UNROUTED
    from .identity_concordance import Ref

    def cells(value_ids: Any) -> tuple:
        return tuple(
            item for item in (cell(value_id) for value_id in value_ids)
            if item is not None
        )

    kind = _control_block_kind(block)
    if kind is None:
        return None
    owner = reason = predicate = callsite = None
    carried: tuple = ()
    sites: tuple = ()
    regions: tuple = ()
    extra: tuple = ()
    cases: tuple = ()
    diagnostics: tuple = ()
    if isinstance(block, StatementBlock):
        region = _region_marker(block)
        callsite_ordinal = _callsite_marker(block)
        if region is not None:
            owner = region_cell(region)
            regions = (owner,) if owner is not None else ()
            extra = ("region", int(region))
            if owner is None:
                reason = REGION_CELL_UNROUTED
        elif callsite_ordinal is not None:
            owner = callsite = callsite_cell(callsite_ordinal)
            extra = ("callsite", int(callsite_ordinal))
            if owner is None:
                reason = REGION_CELL_UNROUTED
        else:
            extra = ("lines", tuple(block.lines))
    elif isinstance(block, ConditionalBlock):
        owner = cell(block.source_node_id)
        predicate = cell(block.predicate_value_id)
        field_cells = tuple(block.carried_field_cells)
        carried = tuple(
            item for item in (
                field_cells[index][2]
                if index < len(field_cells)
                and isinstance(field_cells[index], tuple)
                and len(field_cells[index]) == 3
                and isinstance(field_cells[index][2], Ref)
                else cell(alias[3])
                for index, alias in enumerate(block.carried_aliases)
            )
            if item is not None
        )
        extra = (bool(block.expect_true),)
    elif isinstance(block, LoopBlock):
        owner = cell(block.source_loop_node_id)
        carried = cells(updated for updated, _initial in block.carried_aliases)
        sites = cells(block.control_site_ids)
        extra = (
            str(block.induction), str(block.start), str(block.stop),
            str(block.step), str(block.comparison),
            str(block.schedule_preference),
        )
    elif isinstance(block, WhileBlock):
        owner = cell(block.source_loop_node_id)
        predicate = cell(block.predicate_value_id)
        carried = cells(updated for updated, _initial in block.carried_aliases)
        sites = cells(block.control_site_ids)
    elif isinstance(block, LoopControlBlock):
        # A return edge is owned by its return site's cell.
        owner = (
            block.return_site_cell
            if isinstance(block.return_site_cell, Ref)
            else cell(block.site_node_id)
        )
        predicate = cell(block.predicate_value_id)
        extra = (str(block.action), block.source_action)
    elif isinstance(block, (CallBlock, DispatchBlock, ExternalReferenceCallBlock)):
        owner = callsite = callsite_cell(block.callsite_id)
        extra = (
            getattr(block, "operation", None),
            getattr(block, "identity", None),
        )
    elif isinstance(block, ResourceScopeBlock):
        owner = cell(block.source_scope_id)
    elif isinstance(block, NativeTryBlock):
        owner = cell(block.source_try_node_id)
        carried = cells(
            value_id
            for handler in block.handlers
            for value_id in handler.exception_type_value_ids
        )
        extra = tuple(
            int(handler.source_handler_node_id)
            for handler in block.handlers
        )
    elif isinstance(block, ValidationBlock):
        owner = predicate = cell(block.predicate_value_id)
        diagnostics = cells(block.diagnostic_value_ids)
        extra = (
            int(block.error_code), bool(block.expect_true),
            block.extraction_identity,
        )
    elif isinstance(block, SequenceMutationBlock):
        # The step cell of a composition step, else the effect node's cell,
        # else the source contract the compiler-authored mutation (a glyph
        # token lookup copy) was made from.
        owner = (
            block.mutation.step_cell
            if isinstance(block.mutation.step_cell, Ref)
            else cell(block.mutation.effect_node_id)
            or (
                block.mutation.source_contract_cell
                if isinstance(block.mutation.source_contract_cell, Ref)
                else None
            )
        )
        extra = (str(block.mutation.operator),)
        # Program: ``s0 = {n, 5}; s0.add(7); xs1 = [6, n, -3];
        # xs1.append(1)`` (fuzz p105_7, the "lexical_reads ... fork" family).
        # Wrong record: a display member's birth is owned by the member's
        # producer (``n``), which is shared by both displays; the set's
        # ``add`` and the list's ``append`` then posted two facts on one row
        # and the second was refused as a revision with no new cause, so the
        # whole program was refused.  Two list displays sharing ``n`` already
        # posted one equal fact there.  Rule: a display member birth's fact
        # is "a birth fed by this producer"; the display's own ``clear``
        # (owned by the display node) says which sequence and kind.
        if block.mutation.display_member is not None:
            extra = ("display_member",)
    elif isinstance(block, ScalarFieldWriteBlock):
        owner = (
            block.field_state_cell
            if isinstance(block.field_state_cell, Ref)
            else cell(block.effect_node_id)
        )
        extra = (str(block.dtype),)
    elif isinstance(block, SequenceQueryBlock):
        owner = cell(block.result_value_id)
        if owner is None:
            owner = cell(block.source_call_node_id)
        extra = (str(block.operation),)
    elif isinstance(block, StreamPublishBlock):
        owner = cell(block.value_id)
        extra = (
            bool(block.final),
            tuple(map(int, block.payload_value_ids)),
        )
    elif isinstance(block, StateMachineTick):
        # Like a conditional: the owner is the dispatch construct's node
        # cell, the predicate is the state selector's value cell.
        owner = cell(block.source_node_id)
        predicate = cell(block.state_value_id)
        cases = cells(
            value_id for value_id in block.case_value_ids
            if value_id is not None
        )
        carried = cells(alias[3] for alias in block.carried_aliases)
        extra = (str(block.state),)
    elif isinstance(block, ParallelDeployment):
        extra = (str(block.schedule_preference),)
    if owner is None and reason is None and not isinstance(
        block, ParallelDeployment,
    ):
        reason = CONTROL_OWNER_UNKNOWN
    return kind, owner, reason, (
        predicate, carried, sites, regions, callsite, extra, cases,
        diagnostics,
    )


def control_block_cell(book: Any, scope: Any, block: Any) -> Any:
    """The ``control_block`` cell ``post_control_program`` posted for
    ``block`` under read scope ``scope``, or None.  Read-only: ids resolve
    through ``canonical_value`` rows (the post-relabel arm of
    ``node_identity_cell``) and ``call_binding`` rows, and nothing is
    posted.  A block whose owner is the program-cell stand-in has no cell
    of its own here (None): several ownerless blocks share that row."""

    from .concordance_declarations import CALL_BINDING, CANONICAL_VALUE, CONTROL_BLOCK

    if book is None or scope is None or block is None:
        return None
    scope = tuple(scope)

    def cell(value_id: Any) -> Any:
        if value_id is None or isinstance(value_id, bool) or not isinstance(value_id, int):
            return None
        return book.latest_ref(CANONICAL_VALUE, (scope, int(value_id)))

    def callsite_cell(callsite_id: Any) -> Any:
        if callsite_id is None:
            return None
        bound = book.latest_ref(CALL_BINDING, (scope, int(callsite_id)))
        return bound if bound is not None else cell(callsite_id)

    described = _describe_control_block(
        block, cell=cell, region_cell=lambda _ordinal: None,
        callsite_cell=callsite_cell,
    )
    if described is None or described[1] is None:
        return None
    kind, owner, _reason, _fields = described
    return book.latest_ref(CONTROL_BLOCK, (scope, kind, owner))


def post_control_program(
    graph: Any,
    program: "ControlProgram | None",
    *,
    stage: Any,
    cause: tuple = (),
    label: Any = None,
    previous: "ControlProgram | None" = None,
) -> Any:
    """Post one ControlProgram tree on the book; return its program cell.

    ``graph`` is the ProcessGraph (or its networkx graph) the program was
    built for: its ``lexical_read_scope`` keys every row, its nodes resolve
    the blocks' ids to cells (``node_identity_cell``), its
    ``planning_scope`` resolves region markers to ``deployment_region``
    cells and ``call_binding`` resolves callsites.  ``cause`` names the
    cells whose change made the rewriter return a new tree.  ``label`` is
    the ``control_program`` row's label: ``SHELL`` for the function's one
    program, else the owning construct's cell (derived from the first
    conditional or loop of the root when None).  ``previous`` is the tree
    this one replaces; a block of it that is gone has its placement
    withdrawn by ``Unresolved(COLLAPSED_EMPTY_CONSTRUCT)`` -- an edge, never
    an erasure.

    Rows, per plan 100 section 2.3: each block posts ``control_block``
    CONCORD DERIVED(owner cell, the cells its fact names, ``cause``) -- a
    block whose identity fields changed (a conditional whose carries were
    enriched) posts a REVISE with those cells as its cause, or
    ``Unsourced(CONTROL_BLOCK_REVISION_UNCAUSED)`` when the api would admit
    no cause; a block whose owner cannot be named is keyed on the program
    cell and ``Unsourced(CONTROL_OWNER_UNKNOWN)`` (a region marker with no
    ``deployment_region`` row: ``REGION_CELL_UNROUTED``); two such blocks
    with different facts cannot share the fallback row, so the second is
    not posted and is counted in ``metadata["control_program_unrouted"]``.
    Then ``control_block_placement`` REVISE DERIVED(block cell, parent cell
    or program cell, ``cause``) only when the placement changed, then
    ``control_program`` REVISE when its fact changed.  A rewriter that
    changes nothing posts nothing new.  A graph with no read scope posts one
    ``Unsourced(CONTROL_FUNCTION_SCOPE_UNKNOWN)`` program row under
    ``_UNSCOPED_CONTROL`` so the audit lists the caller.
    """

    if graph is None or program is None:
        return None
    from .concordance_declarations import (
        Arm, CALL_BINDING, COLLAPSED_EMPTY_CONSTRUCT, CONTROL_BLOCK,
        CONTROL_BLOCK_PLACEMENT, CONTROL_PROGRAM, ControlBlockFact, ControlProgramFact,
        DEPLOYMENT_REGION, Placement, REGION_CELL_UNROUTED, ROOT, SHELL,
        SCOPE_REGISTRY,
    )
    from .identity_concordance import (
        Derived, Mode, Ref, Unresolved, current_identity_book,
    )
    from ..common.tensors.topological_reducer import node_identity_cell

    G = getattr(graph, "G", graph)
    metadata = getattr(G, "graph", None) or {}
    book = current_identity_book()
    # A function graph is keyed by its lexical read scope; the module's own
    # graph (the root deployment, no function) by its ingestion scope, the
    # scope its nodes' rows already live under.
    scope = (
        metadata.get("lexical_read_scope")
        or metadata.get("ingestion_value_scope")
    )
    if scope is None:
        raise ValueError(
            f"control_program: no source cell for "
            f"{str(metadata.get('function_name'))!r} -- the graph has no "
            "lexical_read_scope"
        )
    scope = tuple(scope)
    resolvable = bool(
        metadata.get("canonical_value_ids")
        or metadata.get("operand_position_scope")
        or metadata.get("ingestion_value_scope")
    )
    value_to_node: dict[int, Any] = {}
    for node_id, data in G.nodes(data=True):
        value_id = data.get("value_id", node_id)
        if isinstance(value_id, int) and not isinstance(value_id, bool):
            value_to_node.setdefault(int(value_id), node_id)
    planning_scope = metadata.get("planning_scope")
    cause = tuple(dict.fromkeys(item for item in cause if isinstance(item, Ref)))

    def cell(value_id: Any) -> Any:
        if value_id is None or not resolvable or isinstance(value_id, bool):
            return None
        if not isinstance(value_id, int):
            return None
        node_id = value_to_node.get(int(value_id), int(value_id))
        if node_id not in G:
            return None
        return node_identity_cell(graph, node_id)

    def cells(value_ids: Any) -> tuple:
        return tuple(
            item for item in (cell(value_id) for value_id in value_ids)
            if item is not None
        )

    def region_cell(ordinal: Any) -> Any:
        if planning_scope is None or ordinal is None:
            return None
        return book.latest_ref(DEPLOYMENT_REGION, (planning_scope, int(ordinal)))

    def callsite_cell(callsite_id: Any) -> Any:
        if callsite_id is None:
            return None
        bound = book.latest_ref(CALL_BINDING, (scope, int(callsite_id)))
        return bound if bound is not None else cell(callsite_id)

    def describe(block: Any) -> Any:
        return _describe_control_block(
            block, cell=cell, region_cell=region_cell,
            callsite_cell=callsite_cell,
        )

    def fact_cells(fields: tuple) -> tuple:
        (
            predicate, carried, sites, regions, callsite, _extra, cases,
            diagnostics,
        ) = fields
        return tuple(
            item for item in (
                predicate, *carried, *sites, *regions, callsite, *cases,
                *diagnostics,
            )
            if isinstance(item, Ref)
        )

    def admissible_revision(page: Any, row: tuple, sources: tuple) -> bool:
        """The api's REVISE rule, checked at the writer (plan 80 R5)."""

        latest = book.latest_ref(page, row)
        if latest is None:
            return True
        stored = book.pages[page.name]
        newest = max(
            stored.stamp_at(row, column) for column, _fact in stored.history(row)
        )
        previous_sources = {
            source.key for source, _stage in book.edges_into(latest)
        }
        return (
            any(book.stamp_of(source) > newest for source in sources)
            or {source.key for source in sources} != previous_sources
        )

    def revise(page: Any, row: tuple, fact: Any, sources: tuple) -> Any:
        sources = tuple(dict.fromkeys(item for item in sources if isinstance(item, Ref)))
        if not (sources and admissible_revision(page, row, sources)):
            raise ValueError(
                f"{page.name}: no source cell for {row!r} -- the revision "
                "names no cause the api would admit"
            )
        return book.post(
            page, row, fact, stage=stage, provenance=Derived(sources),
            mode=Mode.REVISE,
        )

    def latest_fact(page: Any, row: tuple) -> Any:
        stored = book.pages.get(page.name)
        return None if stored is None else stored.latest(row)

    root_blocks = _flatten_control_sequence(program.root)
    descriptions = {id(block): describe(block) for block in root_blocks}
    if label is None:
        label = SHELL
        for block in root_blocks:
            described = descriptions[id(block)]
            if (
                isinstance(block, (ConditionalBlock, LoopBlock, WhileBlock))
                and described is not None and described[1] is not None
            ):
                label = described[1]
                break
    program_row = (scope, label)
    program_cell = book.latest_ref(CONTROL_PROGRAM, program_row)
    if program_cell is None:
        seeds = tuple(dict.fromkeys((
            *cause,
            book.latest_ref(SCOPE_REGISTRY, scope),
            *(
                described[1] for described in descriptions.values()
                if described is not None and described[1] is not None
            ),
        )))
        seeds = tuple(seed for seed in seeds if isinstance(seed, Ref))
        program_cell = book.post(
            CONTROL_PROGRAM, program_row,
            ControlProgramFact((), (), (), None, (), ()),
            stage=stage,
            provenance=Derived(seeds), mode=Mode.REVISE,
        )
    unrouted = 0
    posted_block_cells: set = set()

    def post_block(block: Any) -> Any:
        nonlocal unrouted
        described = descriptions.get(id(block))
        if described is None:
            described = describe(block)
        if described is None:
            return None
        kind, owner, reason, fields = described
        fallback = owner is None
        owner = program_cell if fallback else owner
        fact = ControlBlockFact(kind, *fields)
        row = (scope, kind, owner)
        sources = tuple(dict.fromkeys((owner, *fact_cells(fields), *cause)))
        latest = book.latest_ref(CONTROL_BLOCK, row)
        if latest is None:
            return book.post(
                CONTROL_BLOCK, row, fact, stage=stage,
                provenance=Derived(sources), mode=Mode.CONCORD,
            )
        if latest_fact(CONTROL_BLOCK, row) == fact:
            if reason is None:
                # Same fact: the design's no-op that still records its edge.
                return book.post(
                    CONTROL_BLOCK, row, fact, stage=stage,
                    provenance=Derived(sources), mode=Mode.CONCORD,
                )
            return latest
        if fallback:
            # Two ownerless blocks with different facts cannot share the
            # program cell's row; the second is the worklist's.
            unrouted += 1
            return None
        return revise(CONTROL_BLOCK, row, fact, sources)

    def place(
        block_cell: Any,
        parent: Any,
        arm: Any,
        index: int,
        ordinal: int,
        predecessors: tuple,
    ) -> None:
        row = (scope, block_cell)
        fact = Placement(parent, arm, int(ordinal), int(index))
        if latest_fact(CONTROL_BLOCK_PLACEMENT, row) == fact:
            return
        # A placement's ordinal is a graph fact: it is determined by the
        # ordered predecessor blocks in this arm.  Include those exact cells
        # so inserting/reordering siblings revises the placement from the
        # changed prefix instead of producing an uncaused rewrite.
        revise(
            CONTROL_BLOCK_PLACEMENT, row, fact,
            (
                block_cell,
                parent if isinstance(parent, Ref) else program_cell,
                *predecessors,
                *cause,
            ),
        )

    def visit(blocks: tuple, parent: Any, arm: Any, index: int) -> tuple:
        placed = []
        for ordinal, block in enumerate(blocks):
            block_cell = post_block(block)
            if block_cell is None:
                continue
            # Program: ``xs.extend([a, b, b])``.  A repeated display member
            # births twice from one producer: the same block fact on the
            # same row, so the same cell.  A cell has one placement; its
            # first occurrence in this arm holds it.
            if any(block_cell.key == item.key for item in placed):
                continue
            posted_block_cells.add(block_cell.key)
            predecessors = tuple(placed)
            placed.append(block_cell)
            place(
                block_cell, parent, arm, index, ordinal, predecessors,
            )
            for child_arm, child_index, children in _control_block_arms(block):
                visit(children, block_cell, child_arm, child_index)
        return tuple(placed)

    root_cells = visit(root_blocks, ROOT, Arm.ROOT_SEQUENCE, 0)

    if previous is not None:
        def previous_cells(blocks: tuple) -> None:
            for block in blocks:
                described = describe(block)
                if described is None:
                    continue
                kind, owner, _reason, fields = described
                row = (scope, kind, program_cell if owner is None else owner)
                block_cell = book.latest_ref(CONTROL_BLOCK, row)
                if (
                    block_cell is not None
                    and block_cell.key not in posted_block_cells
                    and latest_fact(CONTROL_BLOCK, row) == ControlBlockFact(kind, *fields)
                ):
                    placement_row = (scope, block_cell)
                    placement = book.latest_ref(CONTROL_BLOCK_PLACEMENT, placement_row)
                    if placement is not None and not isinstance(
                        latest_fact(CONTROL_BLOCK_PLACEMENT, placement_row),
                        Unresolved,
                    ):
                        revise(
                            CONTROL_BLOCK_PLACEMENT, placement_row,
                            Unresolved(COLLAPSED_EMPTY_CONSTRUCT, (placement,)),
                            (placement, *cause),
                        )
                for _arm, _index, children in _control_block_arms(block):
                    previous_cells(children)
        previous_cells(_flatten_control_sequence(previous.root))

    def region_or_ordinal(ordinal: Any) -> Any:
        found = region_cell(ordinal)
        return found if found is not None else int(ordinal)

    def cell_or_id(value_id: Any) -> Any:
        found = cell(value_id)
        return found if found is not None else value_id

    program_fact = ControlProgramFact(
        regions=tuple(region_or_ordinal(index) for index in program.region_indices),
        uniforms=tuple(program.uniforms),
        value_aliases=tuple(
            (cell_or_id(updated), cell_or_id(initial))
            for updated, initial in program.value_aliases
        ),
        anchor_region=(
            None if program.anchor_region is None
            else region_or_ordinal(program.anchor_region)
        ),
        specialized_conditionals=tuple(
            cell_or_id(node_id)
            for node_id in program.specialized_conditional_node_ids
        ),
        root_blocks=root_cells,
    )
    incumbent_program = latest_fact(CONTROL_PROGRAM, program_row)
    if incumbent_program != program_fact:
        # A rewrite derives from what it rewrote as well as from what it
        # made: the incumbent program cell and the root block cells it
        # held (a rewrite that only removes blocks has no new cell of its
        # own; the removed blocks are its cause).
        incumbent_cells = (
            () if not isinstance(incumbent_program, ControlProgramFact)
            else tuple(
                item for item in incumbent_program.root_blocks
                if isinstance(item, Ref)
            )
        )
        program_cell = revise(
            CONTROL_PROGRAM, program_row, program_fact,
            (
                *root_cells, *cause,
                *(item for item in program_fact.regions if isinstance(item, Ref)),
                book.latest_ref(CONTROL_PROGRAM, program_row),
                *incumbent_cells,
            ),
        )
    if unrouted:
        metadata["control_program_unrouted"] = (
            int(metadata.get("control_program_unrouted") or 0) + unrouted
        )
    return program_cell


def post_control_rewrite(
    graph: Any,
    program: "ControlProgram | None",
    *,
    cause: tuple = (),
    label: Any = None,
    previous: "ControlProgram | None" = None,
) -> Any:
    """``post_control_program`` at stage ``CONTROL_PROGRAM_REWRITE`` with the
    shell label: the one call each rewriter of ``_class_surface_ssa_program``
    makes where it replaces the ControlProgram."""

    from .concordance_declarations import CONTROL_PROGRAM_REWRITE, SHELL

    return post_control_program(
        graph, program, stage=CONTROL_PROGRAM_REWRITE, cause=cause,
        label=SHELL if label is None else label, previous=previous,
    )


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
    if isinstance(block, (DispatchBlock, ResourceScopeBlock, NativeTryBlock)):
        raise ValueError(
            "dispatcher coordination must lower through SSA and a backend "
            "synchronization implementation; source rendering cannot erase it"
        )
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
    if isinstance(block, ScalarFieldWriteBlock):
        suffix = "" if target in {ControlTarget.PYTHON, ControlTarget.FORTRAN} else ";"
        destination_id = (
            int(block.effect_node_id)
            if block.field_value_id is None
            else int(block.field_value_id)
        )
        return (
            f"value_{destination_id} = "
            f"{_render_expression(block.value_expression, target)}{suffix}",
        )
    if isinstance(block, SequenceQueryBlock):
        if block.operation == "truth":
            return (
                f"value_{int(block.result_value_id)} = "
                f"(turing_sequence_length(value_{int(block.sequence_value_id)}) > 0);",
            )
        if block.operation == "length":
            return (
                f"value_{int(block.result_value_id)} = "
                f"turing_sequence_length(value_{int(block.sequence_value_id)});",
            )
        if block.operation == "lookup":
            keys = ", ".join(
                f"value_{int(value_id)}" for value_id in block.key_value_ids
            )
            return (
                f"value_{int(block.result_value_id)} = "
                f"turing_sequence_lookup(value_{int(block.sequence_value_id)}, "
                f"{keys});",
            )
        if block.operation == "sum":
            start = (
                "0" if block.default_value_id is None
                else f"value_{int(block.default_value_id)}"
            )
            return (
                f"value_{int(block.result_value_id)} = "
                f"turing_sequence_sum(value_{int(block.sequence_value_id)}, "
                f"{start});",
            )
        if block.operation == "maximum":
            prefix = ", ".join(
                f"value_{int(value_id)}"
                for value_id in block.reduction_prefix_value_ids
            )
            suffix = ", ".join(
                f"value_{int(value_id)}"
                for value_id in block.reduction_suffix_value_ids
            )
            return (
                f"value_{int(block.result_value_id)} = "
                f"turing_sequence_maximum(value_{int(block.sequence_value_id)}, "
                f"{{{prefix}}}, {{{suffix}}});",
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


def _declared_predicate_regions(block: "ControlBlock") -> tuple[int, ...]:
    """Every predicate-producing region a block's conditionals declare."""

    if isinstance(block, ConditionalBlock):
        return (
            *block.predicate_region_indices,
            *_declared_predicate_regions(block.body),
            *(() if block.orelse is None
              else _declared_predicate_regions(block.orelse)),
        )
    if isinstance(block, SequenceBlock):
        return tuple(
            region for child in block.blocks
            for region in _declared_predicate_regions(child)
        )
    return ()


def order_control_region_dependencies(
    program: "ControlProgram", dependencies: Iterable[tuple[int, int]],
) -> "ControlProgram":
    """Preserve region dependencies when lexical controls become atomic.

    A valid flat order can interleave a loop body with its initial-value
    producer. Replacing the body's first marker by the entire loop then puts
    that producer after its consumer. Move prerequisites before the enclosing
    control, keeping each operation in its original lexical scope. Visit
    prerequisites on demand so unrelated post-loop effects remain post-loop.
    """
    edges = tuple((int(a), int(b)) for a, b in dependencies if a != b)

    def marker_only(block):
        if isinstance(block, StatementBlock):
            return _region_marker(block) is not None
        if isinstance(block, SequenceBlock):
            return bool(block.blocks) and all(marker_only(child)
                                              for child in block.blocks)
        return False

    def cyclic_components(prerequisites):
        """Return SCCs in the child dependency quotient graph."""
        adjacency = {index: set() for index in prerequisites}
        for consumer, producers in prerequisites.items():
            for producer in producers:
                adjacency[producer].add(consumer)
        next_index = 0
        indices, lowlinks = {}, {}
        stack, on_stack, found = [], set(), []

        def strongconnect(vertex):
            nonlocal next_index
            indices[vertex] = lowlinks[vertex] = next_index
            next_index += 1
            stack.append(vertex)
            on_stack.add(vertex)
            for successor in adjacency[vertex]:
                if successor not in indices:
                    strongconnect(successor)
                    lowlinks[vertex] = min(
                        lowlinks[vertex], lowlinks[successor])
                elif successor in on_stack:
                    lowlinks[vertex] = min(
                        lowlinks[vertex], indices[successor])
            if lowlinks[vertex] == indices[vertex]:
                component = set()
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.add(member)
                    if member == vertex:
                        break
                if len(component) > 1:
                    found.append(component)

        for vertex in adjacency:
            if vertex not in indices:
                strongconnect(vertex)
        return found

    def sink_into_body(block, additions):
        """Refine one over-coarse loop atom with bracketed region blocks."""
        if not isinstance(block, (LoopBlock, WhileBlock, ResourceScopeBlock)):
            return None
        body = block.body
        body_blocks = body.blocks if isinstance(body, SequenceBlock) else (body,)
        return replace(
            block, body=SequenceBlock((*body_blocks, *additions)))

    def visit(block):
        if isinstance(block, StatementBlock):
            region = _region_marker(block)
            return block, set() if region is None else {region}
        if isinstance(block, SequenceBlock):
            children, memberships = [], []
            for child in block.blocks:
                rewritten, regions = visit(child)
                children.append(rewritten)
                memberships.append(regions)
            owners = {}
            for index, regions in enumerate(memberships):
                for region in regions:
                    if region in owners:
                        raise ValueError(f"region {region} has multiple sequential control owners")
                    owners[region] = index
            prerequisites = {i: set() for i in range(len(children))}
            for producer, consumer in edges:
                left, right = owners.get(producer), owners.get(consumer)
                if left is not None and right is not None and left != right:
                    prerequisites[right].add(left)
            for index, child in enumerate(children):
                for region in _declared_predicate_regions(child):
                    owner = owners.get(int(region))
                    if owner is not None and owner != index:
                        prerequisites[index].add(owner)

            # Collapsing a lexical loop to one child can create a cycle in
            # this quotient even when the underlying region graph is acyclic:
            # an omitted loop-owned region B appears beside a loop containing
            # A and C while the real order is A -> B -> C.  B must execute per
            # iteration, between A and C.  Refine the over-coarse loop atom by
            # sinking marker-only numerical regions into its body, then order
            # that body with the original dependency graph.  Controls and
            # effectful blocks are never guessed into a new lexical scope.
            for component in cyclic_components(prerequisites):
                containers = [
                    index for index in component
                    if len(memberships[index]) > 1
                    and isinstance(children[index], (
                        LoopBlock, WhileBlock, ResourceScopeBlock))
                ]
                movable = sorted(component - set(containers))
                if (len(containers) == 1 and movable
                        and all(marker_only(children[index])
                                for index in movable)):
                    container = containers[0]
                    refined = sink_into_body(
                        children[container],
                        tuple(children[index] for index in movable),
                    )
                    if refined is not None:
                        rewritten_children = [
                            child for index, child in enumerate(children)
                            if index not in movable
                        ]
                        rewritten_container = container - sum(
                            index < container for index in movable)
                        rewritten_children[rewritten_container] = refined
                        return visit(SequenceBlock(tuple(rewritten_children)))
            ordered, active, complete = [], set(), set()

            def schedule(index):
                if index in complete:
                    return
                if index in active:
                    raise ValueError(
                        "region dependencies cross atomic control boundaries cyclically: "
                        f"regions={sorted(set().union(*(memberships[i] for i in active)))}"
                    )
                active.add(index)
                for predecessor in sorted(prerequisites[index]):
                    schedule(predecessor)
                active.remove(index)
                complete.add(index)
                ordered.append(children[index])

            for index in range(len(children)):
                schedule(index)
            return replace(block, blocks=tuple(ordered)), set(owners)
        fields = (
            ("condition", "body") if isinstance(block, WhileBlock)
            else ("body", "orelse") if isinstance(block, ConditionalBlock)
            else ("body",) if isinstance(block, (LoopBlock, ResourceScopeBlock))
            else ("callee",) if isinstance(block, CallBlock)
            else ()
        )
        if isinstance(block, NativeTryBlock):
            body, body_regions = visit(block.body)
            handlers = []
            regions = set(body_regions)
            for handler in block.handlers:
                handler_body, handler_regions = visit(handler.body)
                regions.update(handler_regions)
                handlers.append(replace(handler, body=handler_body))
            orelse = None
            if block.orelse is not None:
                orelse, members = visit(block.orelse)
                regions.update(members)
            finalbody, members = visit(block.finalbody)
            regions.update(members)
            return replace(
                block, body=body, handlers=tuple(handlers),
                orelse=orelse, finalbody=finalbody,
            ), regions
        changes, regions = {}, set()
        for field in fields:
            child = getattr(block, field)
            if child is not None:
                changes[field], members = visit(child)
                regions.update(members)
        return replace(block, **changes) if changes else block, regions

    root, _regions = visit(program.root)
    return replace(program, root=root)


def place_loop_carried_region_producers(
    program: "ControlProgram",
    region_outputs: Mapping[int, Iterable[int]],
    value_aliases: Mapping[int, int] | None = None,
    lexical_read_scope: Any = None,
) -> "ControlProgram":
    """Put an exact loop-carried producer in the loop that carries it.

    Region scheduling and lexical-control composition are separate stages.
    The latter can therefore replace the first region marker in a loop with
    the whole loop while leaving the region that publishes the loop's updated
    value beside it.  ``LoopBlock.carried_aliases`` and ``result_ports`` are
    the durable identity receipt proving that publication belongs on the
    latch.  Consume that receipt here; never ask SSA lowering to invent a
    producer for a value whose real region was left outside the loop.

    Only marker-only siblings move.  Moving another control or an effectful
    block would guess at lexical semantics and remains deliberately refused.
    """

    aliases = {
        int(alias): int(resident)
        for alias, resident in (value_aliases or {}).items()
    }

    def resident(value_id: int) -> int:
        """Return the concorded storage identity for one planning value."""

        current = int(value_id)
        path: set[int] = set()
        while current in aliases and aliases[current] != current:
            if current in path:
                raise ValueError(
                    "cyclic loop-region identity concordance: "
                    f"{tuple((*path, current))!r}"
                )
            path.add(current)
            current = int(aliases[current])
        return current

    outputs = {
        int(region): frozenset(resident(value) for value in values)
        for region, values in region_outputs.items()
    }

    def carried_updates(block: ControlBlock) -> frozenset[int]:
        if not isinstance(block, (LoopBlock, WhileBlock)):
            return frozenset()
        return frozenset((
            *(resident(updated) for updated, _initial in block.carried_aliases),
            *(
                resident(updated)
                for _port, _initial, updated in block.result_ports
            ),
        ))

    def marker_regions(block: ControlBlock) -> frozenset[int]:
        if isinstance(block, StatementBlock):
            region = _region_marker(block)
            return frozenset() if region is None else frozenset((region,))
        if isinstance(block, SequenceBlock):
            found: set[int] = set()
            for child in block.blocks:
                child_regions = marker_regions(child)
                if not child_regions:
                    return frozenset()
                found.update(child_regions)
            return frozenset(found)
        return frozenset()

    membership_page = None
    if lexical_read_scope is not None:
        from .identity_concordance import current_identity_book

        membership_page = current_identity_book().page("loop_region_membership")

    def owned_regions(block: ControlBlock) -> frozenset[int] | None:
        """The regions the composer recorded for this loop, if any.

        The fact is a tuple of ``deployment_region`` cells (their region
        ordinal is ``ref.row[1]``) or, for a graph planned without a
        planning scope, the ordinals themselves.  A loop with no row falls
        back to the marker regions of its body; that fallback is recorded
        first as ``Unresolved(REGION_OWNERSHIP_UNKNOWN)`` reading the loop's
        cell (plan 80, A2.4 / E14).
        """

        loop_node = getattr(block, "source_loop_node_id", None)
        if membership_page is None or loop_node is None:
            return None
        from .identity_concordance import Unresolved as Unresolved_

        row = (tuple(lexical_read_scope), int(loop_node))
        owned = membership_page.latest(row)
        if owned is None:
            from .concordance_declarations import (
                CANONICAL_VALUE, LOOP_COMPOSER, LOOP_REGION_MEMBERSHIP,
                REGION_OWNERSHIP_UNKNOWN,
            )
            from .identity_concordance import (
                Derived, Mode, Unresolved, current_identity_book,
            )

            book = current_identity_book()
            loop_cell = book.latest_ref(CANONICAL_VALUE, row)
            if loop_cell is not None:
                book.post(
                    LOOP_REGION_MEMBERSHIP, row,
                    Unresolved(REGION_OWNERSHIP_UNKNOWN, read=(loop_cell,)),
                    stage=LOOP_COMPOSER, provenance=Derived((loop_cell,)),
                    mode=Mode.REVISE,
                )
            return None
        if isinstance(owned, Unresolved_):
            return None
        return frozenset(
            int(item) if isinstance(item, int) else int(item.row[1])
            for item in owned
        )

    def append_to_body(block: ControlBlock, additions: tuple[ControlBlock, ...]):
        body = block.body
        body_blocks = body.blocks if isinstance(body, SequenceBlock) else (body,)
        return replace(block, body=SequenceBlock((*body_blocks, *additions)))

    def visit(block: ControlBlock) -> ControlBlock:
        if isinstance(block, SequenceBlock):
            children = [visit(child) for child in block.blocks]
            claimed: set[int] = set()
            additions: dict[int, list[ControlBlock]] = {}
            for owner_index, owner in enumerate(children):
                updates = carried_updates(owner)
                if not updates:
                    continue
                owned = owned_regions(owner)
                for candidate_index, candidate in enumerate(children):
                    if candidate_index == owner_index or candidate_index in claimed:
                        continue
                    regions = marker_regions(candidate)
                    if not regions:
                        continue
                    if owned is not None and not regions <= owned:
                        # The book says this loop does not own the region;
                        # a shared arena (``v`` written inside the inner
                        # loop and again after it) is not ownership.
                        continue
                    if any(outputs.get(region, frozenset()) & updates
                           for region in regions):
                        claimed.add(candidate_index)
                        additions.setdefault(owner_index, []).append(candidate)
            rewritten = []
            for index, child in enumerate(children):
                if index in claimed:
                    continue
                if index in additions:
                    child = append_to_body(child, tuple(additions[index]))
                    child = visit(child)
                rewritten.append(child)
            return replace(block, blocks=tuple(rewritten))
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=visit(block.body),
                orelse=None if block.orelse is None else visit(block.orelse),
            )
        if isinstance(block, WhileBlock):
            return replace(
                block, condition=visit(block.condition), body=visit(block.body),
            )
        if isinstance(block, (LoopBlock, ResourceScopeBlock)):
            return replace(block, body=visit(block.body))
        if isinstance(block, NativeTryBlock):
            return replace(
                block, body=visit(block.body),
                handlers=tuple(replace(item, body=visit(item.body)) for item in block.handlers),
                orelse=None if block.orelse is None else visit(block.orelse),
                finalbody=visit(block.finalbody),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=visit(block.callee))
        return block

    return replace(program, root=visit(program.root))


def _anchored_control(control: "ControlProgram") -> bool:
    """A region-less control the overlay must place by anchor, not drop.

    Only source conditionals qualify: a region-less loop is handled by the
    existing sequence-mutation / call-only-induction rule below.
    """

    if control.region_indices:
        return False
    root = control.root
    blocks = root.blocks if isinstance(root, SequenceBlock) else (root,)
    return any(isinstance(block, ConditionalBlock) for block in blocks)


def enrich_represented_conditionals(
    program: "ControlProgram",
    candidates: "Iterable[ControlProgram]",
    *,
    enrich_loop_continuations: bool = True,
) -> tuple["ControlProgram", tuple[dict[str, object], ...]]:
    """Fill missing metadata on conditionals already nested in ``program``.

    Loop composition can preserve a source conditional before the ordinary
    conditional pass discovers its carried aliases.  Source identity proves
    that both blocks describe the same branch, but replacing the loop-owned
    block would discard its lexical body.  Merge only keyed metadata instead:
    the resident entry wins every equal-key tie and a candidate contributes
    only a key the resident does not yet contain.
    """

    from dataclasses import replace

    candidate_by_source: dict[int, ConditionalBlock] = {}
    for candidate in candidates:
        roots = (
            candidate.root.blocks
            if isinstance(candidate.root, SequenceBlock)
            else (candidate.root,)
        )
        for block in roots:
            if (
                isinstance(block, ConditionalBlock)
                and block.source_node_id is not None
            ):
                candidate_by_source.setdefault(int(block.source_node_id), block)
    receipts: list[dict[str, object]] = []

    def can_fall_through(block: ControlBlock | None) -> bool:
        """Whether an arm has a path to its enclosing conditional merge.

        Only an unconditional loop-control instruction closes that path.
        A conditional loop-control instruction retains its untaken path, and
        controls inside a nested loop leave that loop rather than the arm we
        are inspecting.  This is deliberately an existential query: one
        surviving path is enough for branch-carried state to reach the merge.
        """

        if block is None:
            return True
        if isinstance(block, LoopControlBlock):
            return block.predicate_value_id is not None
        if isinstance(block, SequenceBlock):
            return all(can_fall_through(child) for child in block.blocks)
        if isinstance(block, ConditionalBlock):
            return (
                can_fall_through(block.body)
                or can_fall_through(block.orelse)
            )
        if isinstance(block, ResourceScopeBlock):
            return can_fall_through(block.body)
        if isinstance(block, NativeTryBlock):
            return (
                can_fall_through(block.finalbody)
                and (
                    can_fall_through(block.body)
                    or any(can_fall_through(item.body) for item in block.handlers)
                )
            )
        if isinstance(block, StateMachineTick):
            return (
                block.default is None
                or can_fall_through(block.default)
                or any(can_fall_through(body) for _value, body in block.cases)
            )
        # A loop may execute zero times.  Its break/continue instructions are
        # internal edges and therefore do not close the containing arm.
        return True

    def reachable_carried_aliases(
        source_node_id: int,
        field: str,
        proposed: tuple,
        body: ControlBlock,
        orelse: ControlBlock | None,
    ) -> tuple:
        """Discard aliases whose only authored update leaves before merge."""

        body_reaches_merge = can_fall_through(body)
        orelse_reaches_merge = can_fall_through(orelse)
        reachable: list[tuple] = []
        for item in proposed:
            true_value, false_value, initial_value, merged_value = item
            terminal_updates = tuple(
                arm for arm, value, reaches_merge in (
                    ("body", true_value, body_reaches_merge),
                    ("orelse", false_value, orelse_reaches_merge),
                )
                if not reaches_merge and value != initial_value
            )
            reachable_updates = tuple(
                arm for arm, value, reaches_merge in (
                    ("body", true_value, body_reaches_merge),
                    ("orelse", false_value, orelse_reaches_merge),
                )
                if reaches_merge and value != initial_value
            )
            if terminal_updates and not reachable_updates:
                receipt = {
                    "source_conditional_id": int(source_node_id),
                    "field": str(field),
                    "key": merged_value,
                    "incumbent": None,
                    "candidate": item,
                    "outcome": "terminal_only_update_not_merged",
                    "terminal_arms": terminal_updates,
                    "priority": "exact_lexical_fallthrough",
                    "tie_policy": "incumbent",
                }
                if receipt not in receipts:
                    receipts.append(receipt)
                continue
            reachable.append(item)
        return tuple(reachable)

    def merge_keyed(
        source_node_id: int,
        field: str,
        incumbent: tuple,
        proposed: tuple,
        key_index: int,
    ) -> tuple:
        settled = list(incumbent)
        by_key = {item[key_index]: item for item in incumbent}
        for item in proposed:
            key = item[key_index]
            resident = by_key.get(key)
            if resident is None:
                settled.append(item)
                by_key[key] = item
                outcome = "candidate_added"
            elif resident == item:
                outcome = "equal_incumbent_retained"
            else:
                outcome = "conflicting_incumbent_retained"
            receipt = {
                "source_conditional_id": int(source_node_id),
                "field": str(field),
                "key": key,
                "incumbent": resident,
                "candidate": item,
                "outcome": outcome,
                "priority": "exact_source_conditional_identity",
                "tie_policy": "incumbent",
            }
            if receipt not in receipts:
                receipts.append(receipt)
        return tuple(settled)

    def visit(block: ControlBlock) -> ControlBlock:
        if isinstance(block, SequenceBlock):
            return replace(block, blocks=tuple(map(visit, block.blocks)))
        if isinstance(block, ConditionalBlock):
            body = visit(block.body)
            orelse = None if block.orelse is None else visit(block.orelse)
            candidate = (
                None
                if block.source_node_id is None
                else candidate_by_source.get(int(block.source_node_id))
            )
            if candidate is None:
                return replace(block, body=body, orelse=orelse)
            source_node_id = int(block.source_node_id)
            carried_aliases = reachable_carried_aliases(
                source_node_id,
                "carried_aliases",
                candidate.carried_aliases,
                body,
                orelse,
            )
            carried_sequence_aliases = reachable_carried_aliases(
                source_node_id,
                "carried_sequence_aliases",
                candidate.carried_sequence_aliases,
                body,
                orelse,
            )
            return replace(
                block,
                body=body,
                orelse=orelse,
                predicate_expression=(
                    block.predicate_expression
                    if block.predicate_expression is not None
                    else candidate.predicate_expression
                ),
                carried_aliases=merge_keyed(
                    source_node_id, "carried_aliases",
                    block.carried_aliases, carried_aliases, 3,
                ),
                carried_sequence_aliases=merge_keyed(
                    source_node_id, "carried_sequence_aliases",
                    block.carried_sequence_aliases,
                    carried_sequence_aliases, 3,
                ),
                result_aliases=merge_keyed(
                    source_node_id, "result_aliases",
                    block.result_aliases, candidate.result_aliases, 2,
                ),
                entry_record_projections=tuple(dict.fromkeys((
                    *block.entry_record_projections,
                    *candidate.entry_record_projections,
                ))),
                body_callsite_ids=tuple(dict.fromkeys((
                    *block.body_callsite_ids, *candidate.body_callsite_ids,
                ))),
                orelse_callsite_ids=tuple(dict.fromkeys((
                    *block.orelse_callsite_ids,
                    *candidate.orelse_callsite_ids,
                ))),
            )
        if isinstance(block, WhileBlock):
            return replace(
                block, condition=visit(block.condition), body=visit(block.body),
            )
        if isinstance(block, (LoopBlock, ResourceScopeBlock)):
            return replace(block, body=visit(block.body))
        if isinstance(block, NativeTryBlock):
            return replace(
                block, body=visit(block.body),
                handlers=tuple(replace(item, body=visit(item.body)) for item in block.handlers),
                orelse=None if block.orelse is None else visit(block.orelse),
                finalbody=visit(block.finalbody),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=visit(block.callee))
        if isinstance(block, StateMachineTick):
            return replace(
                block,
                cases=tuple((value, visit(body)) for value, body in block.cases),
                default=(
                    None if block.default is None else visit(block.default)
                ),
            )
        if isinstance(block, ParallelDeployment):
            return replace(block, lanes=tuple(map(visit, block.lanes)))
        return block

    def conditional_aliases(block: ControlBlock):
        """Yield aliases in this loop body, excluding nested loop scopes."""

        if isinstance(block, ConditionalBlock):
            yield from block.carried_aliases
            yield from conditional_aliases(block.body)
            if block.orelse is not None:
                yield from conditional_aliases(block.orelse)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                yield from conditional_aliases(child)
        elif isinstance(block, ResourceScopeBlock):
            yield from conditional_aliases(block.body)
        elif isinstance(block, NativeTryBlock):
            yield from conditional_aliases(block.body)
            for handler in block.handlers:
                yield from conditional_aliases(handler.body)
            if block.orelse is not None:
                yield from conditional_aliases(block.orelse)
            yield from conditional_aliases(block.finalbody)
        elif isinstance(block, StateMachineTick):
            # Loop-carry enrichment reads only (initial, merged); a tick's
            # N-way alias is reported in the conditional's 4-tuple shape.
            for arm_ids, default_id, initial_id, merged_id in (
                block.carried_aliases
            ):
                yield (
                    int(arm_ids[0]) if arm_ids else int(default_id),
                    int(default_id), int(initial_id), int(merged_id),
                )
            for _value, body in block.cases:
                yield from conditional_aliases(body)
            if block.default is not None:
                yield from conditional_aliases(block.default)
        elif isinstance(block, ParallelDeployment):
            for lane in block.lanes:
                yield from conditional_aliases(lane)

    def enrich_loop_carries(block: ControlBlock) -> ControlBlock:
        if isinstance(block, SequenceBlock):
            return replace(
                block,
                blocks=tuple(enrich_loop_carries(child)
                             for child in block.blocks),
            )
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=enrich_loop_carries(block.body),
                orelse=(
                    None if block.orelse is None
                    else enrich_loop_carries(block.orelse)
                ),
            )
        if isinstance(block, (LoopBlock, WhileBlock)):
            body = enrich_loop_carries(block.body)
            condition = (
                enrich_loop_carries(block.condition)
                if isinstance(block, WhileBlock) else None
            )
            successors: dict[int, set[int]] = {}
            for _true, _false, initial, merged in conditional_aliases(body):
                if int(initial) != int(merged):
                    successors.setdefault(int(initial), set()).add(int(merged))
            snapshot_updates = {
                int(updated)
                for _port, _initial, updated in block.result_ports
            }
            incumbent = list(block.carried_aliases)
            additions: list[tuple[int, int]] = []
            for updated, initial in incumbent:
                updated = int(updated)
                initial = int(initial)
                if updated not in snapshot_updates:
                    continue
                chain = [updated]
                seen = {updated}
                while True:
                    choices = tuple(sorted(successors.get(chain[-1], ())))
                    if not choices:
                        break
                    if len(choices) != 1:
                        receipt = {
                            "source_loop_node_id": block.source_loop_node_id,
                            "snapshot_updated_value_id": updated,
                            "initial_value_id": initial,
                            "candidate_value_ids": choices,
                            "outcome": "ambiguous_incumbent_retained",
                            "priority": "conditional_continuation_chain",
                            "tie_policy": "incumbent",
                        }
                        if receipt not in receipts:
                            receipts.append(receipt)
                        chain = [updated]
                        break
                    successor = int(choices[0])
                    if successor in seen:
                        receipt = {
                            "source_loop_node_id": block.source_loop_node_id,
                            "snapshot_updated_value_id": updated,
                            "initial_value_id": initial,
                            "cycle_value_id": successor,
                            "outcome": "cyclic_incumbent_retained",
                            "priority": "conditional_continuation_chain",
                            "tie_policy": "incumbent",
                        }
                        if receipt not in receipts:
                            receipts.append(receipt)
                        chain = [updated]
                        break
                    seen.add(successor)
                    chain.append(successor)
                continued = chain[-1]
                pair = (continued, initial)
                if continued == updated or pair in incumbent or pair in additions:
                    continue
                additions.append(pair)
                receipt = {
                    "source_loop_node_id": block.source_loop_node_id,
                    "snapshot_updated_value_id": updated,
                    "continued_updated_value_id": continued,
                    "initial_value_id": initial,
                    "continuation_chain": tuple(chain),
                    "outcome": "unique_continuation_added",
                    "priority": "unique_conditional_continuation",
                    "tie_policy": "incumbent",
                }
                if receipt not in receipts:
                    receipts.append(receipt)
            carried_aliases = tuple((*additions, *incumbent))
            if isinstance(block, WhileBlock):
                return replace(
                    block,
                    condition=condition,
                    body=body,
                    carried_aliases=carried_aliases,
                )
            return replace(
                block, body=body, carried_aliases=carried_aliases,
            )
        if isinstance(block, ResourceScopeBlock):
            return replace(block, body=enrich_loop_carries(block.body))
        if isinstance(block, NativeTryBlock):
            return replace(
                block, body=enrich_loop_carries(block.body),
                handlers=tuple(replace(item, body=enrich_loop_carries(item.body)) for item in block.handlers),
                orelse=None if block.orelse is None else enrich_loop_carries(block.orelse),
                finalbody=enrich_loop_carries(block.finalbody),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=enrich_loop_carries(block.callee))
        if isinstance(block, StateMachineTick):
            return replace(
                block,
                cases=tuple((value, enrich_loop_carries(body))
                            for value, body in block.cases),
                default=(
                    None if block.default is None
                    else enrich_loop_carries(block.default)
                ),
            )
        if isinstance(block, ParallelDeployment):
            return replace(
                block,
                lanes=tuple(enrich_loop_carries(lane)
                            for lane in block.lanes),
            )
        return block

    visited_root = visit(program.root)
    enriched_root = (
        enrich_loop_carries(visited_root)
        if enrich_loop_continuations else visited_root
    )
    return replace(program, root=enriched_root), tuple(receipts)


def _insert_before_marker(
    block: "ControlBlock", anchor: int | None, inserted: "ControlBlock",
    encloses: "Callable[[int, int], bool] | None" = None,
) -> tuple["ControlBlock", bool]:
    """Insert ``inserted`` immediately before the ``anchor`` region marker.

    Walks sequences and the bodies of conditionals/loops; returns the
    rewritten block and whether the anchor was found.  ``anchor=None``
    never matches, so the caller appends at the end of the scope instead.

    ``encloses(container, inserted)`` -- source conditional ids, answered
    from the authored source by the caller -- says whether a conditional
    lexically holds the inserted one.  A conditional that holds the anchor
    marker in an arm but does NOT lexically hold the inserted conditional is
    a later sibling: the inserted block goes before that conditional, never
    into its arm.  (``if rejected and allow: rejected = False`` then
    ``if rejected: x = x * 0.5; continue``: the anchor is the first region
    after the guard, which lives in the second conditional's arm.)
    """

    if anchor is None:
        return block, False
    if isinstance(block, SequenceBlock):
        children = []
        found = False
        for child in block.blocks:
            if (
                not found
                and isinstance(child, StatementBlock)
                and _region_marker(child) == anchor
            ):
                children.append(inserted)
                found = True
                children.append(child)
                continue
            if not found:
                child, found = _insert_before_marker(
                    child, anchor, inserted, encloses,
                )
            children.append(child)
        return SequenceBlock(tuple(children)), found
    if isinstance(block, StatementBlock):
        if _region_marker(block) == anchor:
            return SequenceBlock((inserted, block)), True
        return block, False
    if isinstance(block, ConditionalBlock):
        body, found = _insert_before_marker(
            block.body, anchor, inserted, encloses,
        )
        orelse = block.orelse
        if not found and orelse is not None:
            orelse, found = _insert_before_marker(
                orelse, anchor, inserted, encloses,
            )
        if found and encloses is not None and block.source_node_id is not None:
            inserted_ids = tuple(
                int(item.source_node_id)
                for item in (
                    inserted.blocks if isinstance(inserted, SequenceBlock)
                    else (inserted,)
                )
                if isinstance(item, ConditionalBlock)
                and item.source_node_id is not None
            )
            if inserted_ids and not any(
                encloses(int(block.source_node_id), inserted_id)
                for inserted_id in inserted_ids
            ):
                return SequenceBlock((inserted, block)), True
        return replace(block, body=body, orelse=orelse), found
    if isinstance(block, (LoopBlock, WhileBlock)):
        body, found = _insert_before_marker(
            block.body, anchor, inserted, encloses,
        )
        return replace(block, body=body), found
    return block, False


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
    *,
    graph: Any = None,
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
            return replace(
                block,
                body=substitute(block.body),
                orelse=None if block.orelse is None else substitute(block.orelse),
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
                control_site_ids=block.control_site_ids,
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
                control_site_ids=block.control_site_ids,
                carried_seeds=block.carried_seeds,
                recursion_region_id=block.recursion_region_id,
                predicate_expression=block.predicate_expression,
                sequence_mutations=block.sequence_mutations,
                source_loop_node_id=block.source_loop_node_id,
                terminal_controls=block.terminal_controls,
            )
        if isinstance(block, LoopControlBlock):
            return block
        if isinstance(block, (SequenceMutationBlock, ScalarFieldWriteBlock)):
            return block
        if isinstance(block, SequenceQueryBlock):
            return block
        if isinstance(block, StateMachineTick):
            return replace(
                block,
                cases=tuple(
                    (value, substitute(body)) for value, body in block.cases
                ),
                default=(
                    None if block.default is None
                    else substitute(block.default)
                ),
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
        if isinstance(block, ResourceScopeBlock):
            return replace(block, body=substitute(block.body) or SequenceBlock(()))
        if isinstance(block, NativeTryBlock):
            return replace(
                block, body=substitute(block.body) or SequenceBlock(()),
                handlers=tuple(replace(item, body=substitute(item.body) or SequenceBlock(())) for item in block.handlers),
                orelse=None if block.orelse is None else substitute(block.orelse),
                finalbody=substitute(block.finalbody) or SequenceBlock(()),
            )
        if isinstance(block, (ExternalReferenceCallBlock, DispatchBlock)):
            return block
        if isinstance(block, ValidationBlock):
            # Composition substitutes region interiors only; it has no
            # retained-value set.  Which guards survive is decided by
            # ``project_control_regions``, before a shell is composed.
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
    composed = ControlProgram(
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
        induction_offsets=program.induction_offsets,
    )
    # Step 9 (plan 100, 2.3): the composed tree is a revision of the planned
    # one; a caller that hands the graph gets the rows posted here.
    post_control_rewrite(graph, composed, previous=program)
    return composed


def project_control_regions(
    program: ControlProgram,
    retained_region_indices: Iterable[int],
    *,
    retained_value_ids: Iterable[int] | None = None,
    preserve_source_loop_carries: bool = False,
    graph: Any = None,
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

    control_defined_values: set[int] = set()

    def collect_control_definitions(block: ControlBlock) -> None:
        if isinstance(block, ConditionalBlock):
            control_defined_values.update(
                int(alias[3]) for alias in block.carried_aliases
            )
            control_defined_values.update(
                int(alias[3]) for alias in block.carried_sequence_aliases
            )
            control_defined_values.update(
                int(alias[2]) for alias in block.result_aliases
            )
            collect_control_definitions(block.body)
            if block.orelse is not None:
                collect_control_definitions(block.orelse)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                collect_control_definitions(child)
        elif isinstance(block, WhileBlock):
            collect_control_definitions(block.condition)
            collect_control_definitions(block.body)
        elif isinstance(block, (LoopBlock, ResourceScopeBlock)):
            collect_control_definitions(block.body)
        elif isinstance(block, NativeTryBlock):
            collect_control_definitions(block.body)
            for handler in block.handlers:
                collect_control_definitions(handler.body)
            if block.orelse is not None:
                collect_control_definitions(block.orelse)
            collect_control_definitions(block.finalbody)
        elif isinstance(block, CallBlock):
            collect_control_definitions(block.callee)
        elif isinstance(block, StateMachineTick):
            for _value, body in block.cases:
                collect_control_definitions(body)
            if block.default is not None:
                collect_control_definitions(block.default)
        elif isinstance(block, ParallelDeployment):
            for lane in block.lanes:
                collect_control_definitions(lane)

    collect_control_definitions(program.root)

    def value_survives_projection(value_id: int) -> bool:
        return (
            retained_values is None
            or int(value_id) in retained_values
            or int(value_id) in control_defined_values
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
            if (
                body is None and orelse is None
                and not block.result_aliases
                and not block.optional_glyph_carry_cells
            ):
                return None
            return ConditionalBlock(
                block.predicate_value_id,
                body or SequenceBlock(()),
                orelse,
                block.expect_true,
                block.predicate_expression,
                tuple(
                    carried for carried in block.carried_aliases
                    if all(value_survives_projection(value_id)
                           for value_id in carried)
                ),
                block.source_node_id,
                tuple(
                    carried for carried in block.carried_sequence_aliases
                    if all(value_survives_projection(value_id)
                           for value_id in carried)
                ),
                entry_record_projections=block.entry_record_projections,
                body_callsite_ids=block.body_callsite_ids,
                orelse_callsite_ids=block.orelse_callsite_ids,
                result_aliases=block.result_aliases,
                sequence_handle_results=block.sequence_handle_results,
                predicate_region_indices=block.predicate_region_indices,
                # Filtered in lockstep with ``carried_aliases`` above so the
                # index alignment between the two survives the projection.
                carried_field_cells=tuple(
                    cells
                    for carried, cells in zip(
                        block.carried_aliases,
                        tuple(block.carried_field_cells) + (None,) * (
                            len(block.carried_aliases)
                            - len(block.carried_field_cells)
                        ),
                    )
                    if all(value_survives_projection(value_id)
                           for value_id in carried)
                ),
                body_value_expressions=block.body_value_expressions,
                orelse_value_expressions=block.orelse_value_expressions,
                optional_glyph_carry_cells=block.optional_glyph_carry_cells,
                short_circuit_operator=block.short_circuit_operator,
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
                and block.source_loop_node_id is None
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
                    or int(updated) in control_defined_values
                    or (
                        preserve_source_loop_carries
                        and block.source_loop_node_id is not None
                    )
                    or any(
                        int(updated) == int(port_updated)
                        for _port, _init, port_updated in block.result_ports
                    )
                ),
                carried_seeds=block.carried_seeds,
                result_ports=block.result_ports,
                control_site_ids=block.control_site_ids,
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
            # An authored while can own carried scalar updates or ordered
            # operations installed after numeric-region projection. An empty
            # numeric body is not proof that its iterations are unobservable.
            if (body is None and block.source_loop_node_id is None
                    and not block.carried_aliases and not block.sequence_mutations) or (
                condition is None and block.predicate_expression is None
            ):
                return None
            # The test reads its leaves at every header: a carried pair whose
            # initial the predicate reads is retained by the loop itself
            # (``while go: ...; go = second < limit``).
            tested_ids: set[int] = set()
            pending = [block.predicate_expression]
            while pending:
                expression = pending.pop()
                if expression is None:
                    continue
                if expression.op == "value" and expression.value_id is not None:
                    tested_ids.add(int(expression.value_id))
                pending.extend(expression.operands)
            return WhileBlock(
                block.predicate_value_id,
                condition or SequenceBlock(()),
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
                    or int(initial) in tested_ids
                    or int(updated) in retained_values
                    or int(updated) in control_defined_values
                    or (
                        preserve_source_loop_carries
                        and block.source_loop_node_id is not None
                    )
                    or any(
                        int(updated) == int(port_updated)
                        for _port, _init, port_updated in block.result_ports
                    )
                ),
                carried_seeds=block.carried_seeds,
                result_ports=block.result_ports,
                control_site_ids=block.control_site_ids,
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
            literal_ids = tuple(block.case_value_ids) + (None,) * (
                len(block.cases) - len(block.case_value_ids)
            )
            kept = [
                (value, projected, literal_id)
                for (value, body), literal_id in zip(block.cases, literal_ids)
                if (projected := project(body)) is not None
            ]
            cases = tuple((value, projected) for value, projected, _ in kept)
            default = (
                None if block.default is None else project(block.default)
            )
            return (
                replace(
                    block, cases=cases, default=default,
                    case_value_ids=(
                        tuple(literal_id for _, _, literal_id in kept)
                        if block.case_value_ids else ()
                    ),
                )
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
        if isinstance(block, ResourceScopeBlock):
            return replace(block, body=project(block.body) or SequenceBlock(()))
        if isinstance(block, NativeTryBlock):
            return replace(
                block, body=project(block.body) or SequenceBlock(()),
                handlers=tuple(replace(item, body=project(item.body) or SequenceBlock(())) for item in block.handlers),
                orelse=None if block.orelse is None else project(block.orelse),
                finalbody=project(block.finalbody) or SequenceBlock(()),
            )
        if isinstance(block, DispatchBlock):
            return block
        if isinstance(block, ValidationBlock):
            return block
        if isinstance(block, (SequenceMutationBlock, ScalarFieldWriteBlock)):
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
        elif isinstance(block, ConditionalBlock):
            gather_inductions(block.body)
            if block.orelse is not None:
                gather_inductions(block.orelse)
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
        elif isinstance(block, ResourceScopeBlock):
            gather_inductions(block.body)
        elif isinstance(block, NativeTryBlock):
            gather_inductions(block.body)
            for handler in block.handlers:
                gather_inductions(handler.body)
            if block.orelse is not None:
                gather_inductions(block.orelse)
            gather_inductions(block.finalbody)

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
    projected = ControlProgram(
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
        induction_offsets=program.induction_offsets,
    )
    # Step 9 (plan 100, 2.3): a collapsed construct's placement is withdrawn
    # by ``Unresolved(COLLAPSED_EMPTY_CONSTRUCT)`` from ``previous``.
    post_control_rewrite(graph, projected, previous=program)
    return projected


def _order_conditional_state_dependencies(block: ControlBlock) -> ControlBlock:
    """Respect carried state between adjacent conditional regions.

    Flat numeric regions can be independent while their owning branches are
    not: an unselected arm still forwards its incoming state. That dependency
    is declared by carried_aliases and must survive region scheduling. Other
    statements are barriers; this pass never moves a branch across them.
    """
    from dataclasses import replace

    if isinstance(block, ConditionalBlock):
        return replace(
            block, body=_order_conditional_state_dependencies(block.body),
            orelse=(None if block.orelse is None else
                    _order_conditional_state_dependencies(block.orelse)),
        )
    if not isinstance(block, SequenceBlock):
        return block
    children = []
    for child in block.blocks:
        child = _order_conditional_state_dependencies(child)
        children.extend(child.blocks if isinstance(child, SequenceBlock) else (child,))
    start = 0
    while start < len(children):
        if not isinstance(children[start], ConditionalBlock):
            start += 1
            continue
        stop = start + 1
        while stop < len(children) and isinstance(children[stop], ConditionalBlock):
            stop += 1
        run = children[start:stop]
        producers = {int(alias[3]): index for index, child in enumerate(run)
                     for alias in child.carried_aliases}
        dependencies = {
            index: {producers[int(alias[2])] for alias in child.carried_aliases
                    if int(alias[2]) in producers and producers[int(alias[2])] != index}
            for index, child in enumerate(run)
        }
        pending = list(range(len(run)))
        ordered = []
        while pending:
            ready = next((index for index in pending
                          if not dependencies[index].intersection(pending)), None)
            if ready is None:
                raise ValueError("cyclic carried state between sequential conditionals")
            pending.remove(ready)
            ordered.append(run[ready])
        children[start:stop] = ordered
        start = stop
    return replace(block, blocks=tuple(children))


def _markers_within(block: "ControlBlock") -> tuple[int, ...]:
    """Every scheduled region marker a control subtree holds."""

    found: list[int] = []

    def visit(item):
        if isinstance(item, StatementBlock):
            marker = _region_marker(item)
            if marker is not None:
                found.append(int(marker))
            return
        if isinstance(item, SequenceBlock):
            for child in item.blocks:
                visit(child)
        elif isinstance(item, ConditionalBlock):
            visit(item.body)
            if item.orelse is not None:
                visit(item.orelse)
        elif isinstance(item, WhileBlock):
            visit(item.condition)
            visit(item.body)
        elif isinstance(item, LoopBlock):
            visit(item.body)

    visit(block)
    return tuple(found)


def overlay_scheduled_control(
    region_indices: Iterable[int],
    controls: Iterable[ControlProgram],
    *,
    known_nesting: "Mapping[int, Iterable[int]] | None" = None,
    authored_control_sites: "Mapping[tuple[int, int], AuthoredControlSite] | None" = None,
    lexical_encloses: "Callable[[int, int], bool] | None" = None,
    source_graph: Any = None,
    region_source_nodes: Mapping[int, Iterable[int]] | None = None,
) -> ControlProgram:
    """Overlay planned control blocks on the flat scheduled region order.

    ``lexical_encloses(container, inner)``, if given, answers from the
    authored source whether conditional ``container`` lexically holds
    conditional ``inner`` (both source node ids); anchored region-less
    conditionals are then never placed inside a sibling's arm
    (``_insert_before_marker``).

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

    For a declared nested loop with no regions, ``source_graph`` supplies
    the exact authored arm/statement identity. ``region_source_nodes`` is
    the incumbent dispatch region's deployment-node view, used only to
    place that loop among the arm's existing statements. Missing evidence
    refuses instead of running guarded work at the schedule root.
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
    induction_offsets = []
    recursion_regions = []
    deployment_regions = []
    specialized_conditional_node_ids = []
    controls = tuple(controls)
    direct_children = {
        int(parent): frozenset(int(child) for child in children)
        for parent, children in (known_nesting or {}).items()
    }
    authored_control_sites = {
        (int(parent), int(child)): site
        for (parent, child), site in (authored_control_sites or {}).items()
    }
    nested_children_overall = frozenset(
        child
        for children in direct_children.values()
        for child in children
    )

    # A predicate-producing numerical region is a dataflow predecessor, not
    # private syntax owned by one branch. Two sibling structural conditionals
    # can consequently name the same predicate region plus distinct later
    # regions. The shared marker must remain once in the flat dependency
    # schedule; embedding either complete control would otherwise duplicate
    # it, while treating the siblings as nested would invent source structure.
    # Hoist only a marker proven to be a direct Sequence prelude in every
    # claimant. Anything inside a loop/arm remains a real cross-scope overlap
    # and is still refused below.
    initial_sets = tuple(
        frozenset(control.region_indices) for control in controls
    )
    claims = {
        region: tuple(
            index for index, regions in enumerate(initial_sets)
            if region in regions
        )
        for region in set().union(*initial_sets) if initial_sets
    }

    def declared_nested(left: int, right: int) -> bool:
        return (
            left in direct_children.get(right, ())
            or right in direct_children.get(left, ())
            or initial_sets[left] < initial_sets[right]
            or initial_sets[right] < initial_sets[left]
        )

    hoisted = set()
    for region, owners in claims.items():
        if len(owners) < 2 or all(
            declared_nested(left, right)
            for offset, left in enumerate(owners)
            for right in owners[offset + 1:]
        ):
            continue
        if all(
            isinstance(controls[index].root, SequenceBlock)
            and any(
                isinstance(block, StatementBlock)
                and _region_marker(block) == region
                for block in controls[index].root.blocks
            )
            for index in owners
        ):
            hoisted.add(region)
    if hoisted:
        rewritten = []
        for control in controls:
            root = control.root
            if isinstance(root, SequenceBlock):
                root = SequenceBlock(tuple(
                    block for block in root.blocks
                    if not (
                        isinstance(block, StatementBlock)
                        and _region_marker(block) in hoisted
                    )
                ))
            rewritten.append(replace(
                control,
                root=root,
                region_indices=tuple(
                    region for region in control.region_indices
                    if region not in hoisted
                ),
            ))
        controls = tuple(rewritten)

    controlled_sets = tuple(
        frozenset(control.region_indices) for control in controls
    )

    def _declared_descendant(
        child: int, parent: int, visiting: frozenset[int] = frozenset(),
    ) -> bool:
        """Answer containment from the authored control graph alone."""

        if parent in visiting:
            raise ValueError("cyclic declared loop-control containment")
        children = direct_children.get(parent, ())
        return child in children or any(
            _declared_descendant(
                child, intermediate, visiting | {parent},
            )
            for intermediate in children
        )

    def _nested_in(child: int, parent: int) -> bool:
        if child == parent:
            return False
        # The authored control graph is the authority when it relates these
        # controls.  Region sets are only a proxy for otherwise unrelated
        # controls: successive clauses of one comprehension can give the
        # outer clause {r1} and its inner clause {r1, r2}.  Treating the
        # strict subset as reverse containment contradicts the exact
        # generators edge and makes both controls non-maximal.
        if _declared_descendant(child, parent):
            return True
        if _declared_descendant(parent, child):
            return False
        if controlled_sets[child] < controlled_sets[parent]:
            return True
        return False

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
            # Rebuild by `replace` so every field survives embedding --
            # positional reconstruction silently dropped
            # `entry_record_projections` and the arm callsite ownership.
            return (
                replace(
                    block,
                    body=body or SequenceBlock(()),
                    orelse=orelse,
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
                    control_site_ids=block.control_site_ids,
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
                    control_site_ids=block.control_site_ids,
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
                replace(block, cases=tuple(cases), default=default),
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

    def place_authored_comprehension_control(
        root, child_root, authored_site=None,
    ):
        """Consume the declared nesting edge in its exact authored arm.

        In root(value,memo)->visit(item), conditional 11 owns comprehension
        13, but 13 has no numerical regions. The old overlay ignored that
        edge and emitted 13 beside 11, making recursive child 0 a formal.
        Source AST objects select the arm and statement slot; region nodes
        only locate the existing scheduled statements. Neither coordinates
        nor a scalar dtype is evidence of ownership.
        """
        import ast
        from .glsl_deployment_strategy import _source_control_expression

        graph = getattr(source_graph, "G", source_graph)
        if graph is None:
            raise ValueError("region-less nested loop lacks its source graph")

        def expression(node_id):
            if node_id is None:
                return None
            data = graph.nodes.get(int(node_id), {})
            return data.get("authored_expr_obj") or data.get("expr_obj") or (
                _source_control_expression(graph, int(node_id))
            )

        child_expression = expression(
            getattr(child_root, "source_loop_node_id", None)
            if authored_site is None
            else authored_site.child_source_node_id
        )

        def insert_at_comprehension_site(block):
            """Consume an exact comprehension slot without arm inference."""

            if authored_site is None or authored_site.slot not in {
                "iter", "ifs", "elt", "key", "value", "generators",
            }:
                return block, False
            node_id = getattr(block, "source_loop_node_id", None)
            if node_id == authored_site.parent_source_node_id:
                if not isinstance(block, (LoopBlock, WhileBlock)):
                    raise ValueError(
                        "authored comprehension site does not name a loop control"
                    )
                if authored_site.slot == "iter":
                    return SequenceBlock((child_root, block)), True
                if isinstance(block, WhileBlock) and authored_site.slot == "ifs":
                    condition = _flatten_control_sequence(block.condition)
                    return replace(
                        block,
                        condition=SequenceBlock((child_root, *condition)),
                    ), True
                body = _flatten_control_sequence(block.body)
                return replace(
                    block,
                    body=SequenceBlock((child_root, *body)),
                ), True
            if isinstance(block, SequenceBlock):
                found = False
                children = []
                for member in block.blocks:
                    projected, owned = insert_at_comprehension_site(member)
                    if owned and found:
                        raise ValueError(
                            "authored comprehension site has multiple control owners"
                        )
                    found |= owned
                    children.append(projected)
                return replace(block, blocks=tuple(children)), found
            if isinstance(block, ConditionalBlock):
                body, body_found = insert_at_comprehension_site(block.body)
                orelse = block.orelse
                else_found = False
                if orelse is not None:
                    orelse, else_found = insert_at_comprehension_site(orelse)
                return replace(block, body=body, orelse=orelse), (
                    body_found or else_found
                )
            if isinstance(block, (LoopBlock, WhileBlock)):
                body, found = insert_at_comprehension_site(block.body)
                return replace(block, body=body), found
            return block, False

        if authored_site is not None and authored_site.slot in {
            "iter", "ifs", "elt", "key", "value", "generators",
        }:
            projected, found = insert_at_comprehension_site(root)
            if not found:
                raise ValueError(
                    "declared nested loop has no exact authored execution site"
                )
            return projected

        def source_ids(block):
            if isinstance(block, StatementBlock):
                marker = _region_marker(block)
                if marker is not None:
                    return tuple((region_source_nodes or {}).get(marker, ()))
                callsite = _callsite_marker(block)
                return () if callsite is None else (callsite,)
            node_id = getattr(block, "source_node_id", None)
            if node_id is None:
                node_id = getattr(block, "source_loop_node_id", None)
            if node_id is None:
                node_id = getattr(block, "site_node_id", None)
            return () if node_id is None else (node_id,)

        def insert(block):
            if isinstance(block, SequenceBlock):
                found = False
                children = []
                for member in block.blocks:
                    projected, owned = insert(member)
                    if owned and found:
                        raise ValueError("nested loop has multiple authored owners")
                    found |= owned
                    children.append(projected)
                return replace(block, blocks=tuple(children)), found
            parent_id = getattr(block, "source_node_id", None)
            if parent_id is None:
                parent_id = getattr(block, "source_loop_node_id", None)
            parent_expression = expression(parent_id)
            slots = []
            for arm in ("body", "orelse"):
                if not hasattr(block, arm):
                    continue
                if isinstance(parent_expression, ast.BoolOp):
                    # Program: ``... or any(... for block in ... for i in
                    # block.instrs)`` (native_object_requirement_sources).
                    # Wrong record: a short-circuit conditional's source is
                    # the BoolOp, which has no ``body``/``orelse`` statements,
                    # so a loop authored in its guarded operand found no arm.
                    # Rule: the guarded operand is the conditional's arm --
                    # ``body`` for ``and``, ``orelse`` for ``or`` (binary
                    # after source normalization).
                    guarded = (
                        "body" if isinstance(parent_expression.op, ast.And)
                        else "orelse"
                    )
                    statements = (
                        tuple(parent_expression.values[1:])
                        if arm == guarded else ()
                    )
                else:
                    statements = getattr(parent_expression, arm, ())
                if isinstance(statements, ast.AST):
                    statements = (statements,)
                members = tuple(frozenset(map(id, ast.walk(statement)))
                                for statement in statements)
                matches = [i for i, nodes in enumerate(members)
                           if id(child_expression) in nodes]
                if matches:
                    slots.append((arm, matches, members))
            if not slots:
                return block, False
            if len(slots) != 1 or len(slots[0][1]) != 1:
                raise ValueError("nested loop has ambiguous authored arm")
            arm, (ordinal,), members = slots[0]
            siblings = _flatten_control_sequence(getattr(block, arm))
            # Program: ``A or any(... for b in bs for i in b.instrs)``.
            # Wrong record: the short-circuit arm already listed the inner
            # loop's region marker, and inserting the region-less outer loop
            # (which holds that loop) left the marker beside it -- one region
            # with two sequential owners.  Rule: as in marker embedding, a
            # region the inserted control owns is that control's, so its
            # marker leaves the arm.
            owned_regions = set(_markers_within(child_root))
            siblings = [
                sibling for sibling in siblings
                if not (
                    isinstance(sibling, StatementBlock)
                    and _region_marker(sibling) is not None
                    and _region_marker(sibling) in owned_regions
                )
            ]
            offset = len(siblings)
            for i, sibling in enumerate(siblings):
                positions = {j for node in source_ids(sibling)
                             for j, nodes in enumerate(members)
                             if id(expression(node)) in nodes}
                if not positions:
                    raise ValueError("nested loop sibling lacks source identity")
                if min(positions) < ordinal <= max(positions):
                    raise ValueError("scheduled region crosses a nested loop slot")
                if min(positions) >= ordinal:
                    offset = i
                    break
            body = SequenceBlock((*siblings[:offset], child_root, *siblings[offset:]))
            return replace(block, **{arm: body}), True

        projected, found = insert(root)
        if not found:
            raise ValueError("declared nested loop has no exact authored arm")
        return projected

    def nested_root(index: int, visiting=frozenset()) -> ControlBlock:
        if index in nested_roots:
            return nested_roots[index]
        if index in visiting:
            raise ValueError("cyclic loop-control containment")
        root = controls[index].root
        # Region-less children declared nested in this control (a guard
        # clause inside an arm or a loop body) are placed before their
        # anchor region within this root, or appended to it.  They are
        # placed BEFORE the region-owning children are embedded: the anchor
        # is the first region lexically after the child, and when that
        # region belongs to a sibling construct (``if rejected and allow:
        # rejected = False`` followed by ``if rejected: x = x * 0.5;
        # continue`` in a while body) the marker would otherwise already sit
        # inside the sibling's arm, and the child -- whose merge the
        # sibling's test reads -- was nested into that arm.
        children = tuple(direct_children.get(index, ()))
        exact_comprehension_children = sorted(
            (
                child for child in children
                if (site := authored_control_sites.get((index, child)))
                is not None
                and site.slot in {
                    "iter", "ifs", "elt", "key", "value", "generators",
                }
            ),
            key=lambda child: authored_control_sites[(index, child)].path,
            reverse=True,
        )
        remaining_children = sorted(
            set(children).difference(exact_comprehension_children)
        )
        # Exact comprehension sites insert at the front of their execution
        # slot. Consume later-authored siblings first so repeated insertion
        # preserves source evaluation order. A region-owning child normally
        # replaces its markers below. When none of those markers occur in the
        # parent's program, the authored graph edge is the only valid site:
        # this is the nested ``any(... for ...)`` inside an outer
        # comprehension predicate. Place that complete control there rather
        # than pretending its numerical region is a root sibling.
        placed_exact_children = set()
        for child in exact_comprehension_children:
            control = controls[child]
            child_regions = controlled_sets[child]
            present_markers = _marker_scope_paths(root, child_regions)
            if child_regions and present_markers:
                child_root = nested_root(child, visiting | {index})
                root, consumed = embed(root, child_root, child_regions)
                if not consumed:
                    raise ValueError(
                        "authored nested control has no consumable marker at "
                        "its exact parent site"
                    )
                # The exact source edge owns the child's complete region set.
                # A later comprehension clause may have additional regions
                # which cannot appear in the earlier clause's flat body;
                # requiring equality here reverses the graph relationship.
                placed_exact_children.add(child)
                continue
            if isinstance(control.root, (LoopBlock, WhileBlock)) or child_regions:
                root = place_authored_comprehension_control(
                    root,
                    nested_root(child, visiting | {index}),
                    authored_control_sites.get((index, child)),
                )
                placed_exact_children.add(child)
        for child in remaining_children:
            control = controls[child]
            if (not control.region_indices
                    and isinstance(control.root, (LoopBlock, WhileBlock))):
                root = place_authored_comprehension_control(
                    root,
                    nested_root(child, visiting | {index}),
                    authored_control_sites.get((index, child)),
                )
                continue
            if not _anchored_control(control):
                continue
            child_root = nested_root(child, visiting | {index})
            root, found = _insert_before_marker(
                root, control.anchor_region, child_root, lexical_encloses,
            )
            if not found:
                root = SequenceBlock((
                    *(root.blocks if isinstance(root, SequenceBlock) else (root,)),
                    child_root,
                ))
        candidates = [
            child
            for child, child_regions in enumerate(controlled_sets)
            if child != index
            and child not in placed_exact_children
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
                message = (
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
                raise ControlOverlayScopeError(
                    message,
                    control_index=child,
                    region_indices=child_regions,
                    scopes=by_scope,
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
        induction_offsets.extend(control.induction_offsets)
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
            previous = [
                {
                    "index": other,
                    "regions": tuple(controls[other].region_indices),
                    "root": type(controls[other].root).__name__,
                }
                for other in maximal
                if other != index
                and set(controls[other].region_indices) & set(controlled)
            ]
            raise ValueError(
                "maximal control blocks overlap without containment: "
                f"overlap={sorted(overlap)!r}; current={{'index': {index}, "
                f"'regions': {controlled!r}, 'root': "
                f"{type(control.root).__name__!r}}}; previous={previous!r}"
            )
        def owned_regions(owner: int) -> frozenset[int]:
            return controlled_sets[owner] | frozenset().union(*(
                owned_regions(child)
                for child in direct_children.get(owner, ())
            )) if direct_children.get(owner) else controlled_sets[owner]

        complete_controlled = owned_regions(index)
        first = min(complete_controlled, key=positions.__getitem__)
        replacements[first] = nested_root(index)
        covered.update(complete_controlled)
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
    # Program: ``values = [i % 3 for i in range(n)]; seen = {v for v in
    # values}; len(seen)`` (silent lane; the SSA interpreter thread's
    # c471).  Wrong record: the set comprehension's loop has no region of
    # its own (its row is the loop target itself), so it was placed at the
    # schedule root, ahead of the loop that fills ``values``; it read an
    # empty list and ``len`` returned 0 for n=7, where Python gives 3.
    # Rule: a region-less loop over a collection another loop produces runs
    # after that producer loop.
    def producer_control(value_id: int) -> int | None:
        if source_graph is None:
            return None
        graph_view = getattr(source_graph, "G", source_graph)
        data = graph_view.nodes.get(int(value_id)) if int(value_id) in graph_view else None
        if data is None or str(data.get("type")) != "LoopResult":
            return None
        loop_node = next((
            int(parent) for parent, role in data.get("parents") or ()
            if str(role) == "control"
        ), None)
        matches = [
            index for index, control in enumerate(controls)
            if isinstance(control.root, LoopBlock)
            and control.root.source_loop_node_id is not None
            and int(control.root.source_loop_node_id) == loop_node
        ]
        return matches[0] if len(matches) == 1 else None

    after_producer: dict[int, list[int]] = {}
    for index, control in enumerate(controls):
        if (
            control.region_indices
            or index in nested_children_overall
            or not isinstance(control.root, LoopBlock)
            or control.root.source_loop_node_id is None
            or direct_children.get(index)
            or _anchored_control(control)
        ):
            continue
        producers = {
            producer
            for iterable, _target, _induction in control.iterable_bindings
            for producer in (producer_control(int(iterable)),)
            if producer is not None and producer != index
        }
        if len(producers) == 1:
            after_producer.setdefault(producers.pop(), []).append(index)
    deferred = {
        index for indices in after_producer.values() for index in indices
    }
    placed: set[int] = set()

    def with_dependents(index: int, block: ControlBlock) -> list[ControlBlock]:
        """``block`` followed by the region-less loops reading its rows."""

        emitted = [block]
        for dependent in after_producer.get(index, ()):
            if dependent in placed:
                continue
            placed.add(dependent)
            emitted.extend(with_dependents(dependent, nested_root(dependent)))
        return emitted

    blocks = [
        emitted
        for index, block in (
            (index, nested_root(index))
            for index, control in enumerate(controls)
            if not control.region_indices
            and index not in nested_children_overall
            and index not in deferred
            and (
                # An effect-only guard inside an otherwise empty class surface
                # has no scheduled region. Its declared containment edge still
                # owns the guard and return. Materialize that owner so nested_root
                # does not silently discard the child before callsite placement.
                (bool(direct_children.get(index)) and not _anchored_control(control))
                or isinstance(control.root, LoopBlock)
                and (
                    bool(control.root.sequence_mutations)
                    or str(control.root.induction) in call_only_inductions
                    or control.root.source_loop_node_id is not None
                )
                or isinstance(control.root, WhileBlock)
                and bool(control.root.sequence_mutations)
            )
        )
        for emitted in with_dependents(index, block)
    ]
    # Region-less controls that declare an anchor (a guard clause whose arm
    # is only a return/call): those nested under another control were
    # inserted by `nested_root`; the maximal ones are inserted here, just
    # before their anchor region in the flat schedule, or at the end when
    # nothing follows them.
    anchored_top_level: dict[int | None, list[ControlBlock]] = {}
    for index, control in enumerate(controls):
        if index in nested_children_overall or not _anchored_control(control):
            continue
        anchored_top_level.setdefault(
            control.anchor_region, []
        ).append(nested_root(index))
    for region_index in order:
        for anchored in anchored_top_level.pop(region_index, ()):
            blocks.append(anchored)
        replacement = replacements.get(region_index)
        if replacement is not None:
            blocks.append(replacement)
            for producer in after_producer:
                if region_index in controlled_sets[producer]:
                    blocks.extend(with_dependents(producer, None)[1:])
        elif region_index not in covered:
            blocks.append(StatementBlock((
                f"__scheduled_region_{region_index}__",
            )))
    for anchored_blocks in anchored_top_level.values():
        blocks.extend(anchored_blocks)
    # A producer this schedule never embedded leaves its readers where
    # region-less loops always went: the schedule root.
    unplaced = [index for index in sorted(deferred) if index not in placed]
    if unplaced:
        placed.update(unplaced)
        blocks[:0] = [nested_root(index) for index in unplaced]

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
        root=_order_conditional_state_dependencies(SequenceBlock(tuple(blocks))),
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
        induction_offsets=tuple(dict.fromkeys(induction_offsets)),
        specialized_conditional_node_ids=tuple(dict.fromkeys(
            int(node_id) for node_id in specialized_conditional_node_ids
        )),
    )


def place_validations_after_region_producers(
    program: ControlProgram,
    validations: Iterable[ValidationBlock],
    *,
    predicate_regions: Mapping[int, int],
    merged_predicates: frozenset[int] = frozenset(),
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
    # A predicate that a ConditionalBlock merge defines (a short-circuit
    # operator's result) is placed directly after that conditional.
    pending_by_merge: dict[int, list[ValidationBlock]] = {}
    prelude: list[ValidationBlock] = []
    for validation in validations:
        if int(validation.predicate_value_id) in merged_predicates:
            pending_by_merge.setdefault(
                int(validation.predicate_value_id), [],
            ).append(validation)
            continue
        region = predicate_regions.get(int(validation.predicate_value_id))
        if region is None:
            prelude.append(validation)
        else:
            pending_by_region.setdefault(int(region), []).append(validation)
    placed_merges: set[int] = set()

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
            children: list[ControlBlock] = []
            for child in block.blocks:
                children.append(visit(child))
                if isinstance(child, ConditionalBlock):
                    for alias in child.result_aliases:
                        merged = int(alias[2])
                        if merged in pending_by_merge and merged not in placed_merges:
                            children.extend(pending_by_merge[merged])
                            placed_merges.add(merged)
            return replace(block, blocks=tuple(children))
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
    if isinstance(program.root, ConditionalBlock):
        trailing = [
            validation
            for alias in program.root.result_aliases
            if int(alias[2]) in pending_by_merge
            and int(alias[2]) not in placed_merges
            for validation in pending_by_merge[int(alias[2])]
        ]
        placed_merges.update(
            int(alias[2]) for alias in program.root.result_aliases
        )
        if trailing:
            root = SequenceBlock((root, *trailing))
    missing = set(pending_by_region) - placed
    if missing:
        raise ValueError(
            "validation predicate producer regions are absent from the "
            f"scheduled control tree: regions={sorted(missing)!r}"
        )
    missing_merges = set(pending_by_merge) - placed_merges
    if missing_merges:
        raise ValueError(
            "validation predicates name conditional merges absent from the "
            f"control tree's sequences: merges={sorted(missing_merges)!r}"
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
    "control_callsite_markers",
    "ControlDeploymentLane",
    "ControlDeploymentRegion",
    "ControlExpression",
    "ControlSequenceMutation",
    "ScalarFieldWriteBlock",
    "CallBlock",
    "ExternalReferenceCallBlock",
    "NativeExceptionHandler",
    "NativeTryBlock",
    "DispatchBlock",
    "ResourceScopeBlock",
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
    "validation_diagnostic_value_ids",
]
