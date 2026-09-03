"""Lower an existing Turing numerical precompile and control plan into SSA."""

from __future__ import annotations

import ast
import copy
import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import networkx as nx

from .control_source import (
    CallBlock,
    ConditionalBlock,
    ControlBlock,
    ControlExpression,
    ControlProgram,
    ControlSequenceMutation,
    ExternalReferenceCallBlock,
    LoopControlBlock,
    LoopBlock,
    ParallelDeployment,
    SequenceBlock,
    SequenceMutationBlock,
    SequenceQueryBlock,
    StateMachineTick,
    StatementBlock,
    StreamPublishBlock,
    ValidationBlock,
    WhileBlock,
    control_dependency_value_ids,
)
from .deployment_frame import DeploymentJoin
from .hierarchical_plan import (
    PlanCall,
    PlanClosure,
    PlanLine,
    plan_region_to_ssa_instrs,
)
from .precompile_ssa_validator import (
    PrecompileSSAValidationResult,
    ssa_handler_for_precompile,
    validate_precompile_ssa_compatibility,
)
from .ssa_features import XOROSHIRO128SS_FILL, link_required_ssa_features
from ..common.tensors.fused_ir import (
    FusedProgram,
    Meta,
    OpStep,
    canonicalize_elementwise_steps,
)
from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    LLVM_SSA_MODULE,
    translations_for_operation,
)
from ..common.tensors.accelerator_backends.llvm_repository_ssa import (
    import_llvm_to_repository_ssa,
)
from ..transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSAClassDefinition,
    SSAClassField,
    SSAClassMethod,
    SSAClassTable,
    SSAChildTablePoolDescriptor,
    SSADeploymentLane,
    SSADeploymentRegion,
    SSASequenceCapacityPolicy,
    SSASequenceDescriptor,
    SSASequenceTable,
    SSAValue,
)
from ..transmogrifier.ssa_registry import Handler


_REGION_MARKER = re.compile(r"^__scheduled_region_(\d+)__$")
# A CALL STATEMENT. Historically the control program had no vocabulary for an
# authored call -- calls were an overlay stitched into the lowered SSA
# afterwards by lexical anchors -- so a loop body whose only content is a call
# was, in the plan's own language, empty. This marker makes a callsite a
# schedulable statement exactly the way a region is one.
_CALLSITE_MARKER = re.compile(r"^__plan_callsite_(\d+)__$")


@dataclass(frozen=True, order=True)
class SSALoweringShortfall:
    domain: str
    name: str
    location: str
    reason: str


@dataclass(frozen=True, order=True)
class SSACycle:
    function: str
    blocks: tuple[str, ...]
    back_edges: tuple[tuple[str, str], ...]
    phi_blocks: tuple[str, ...]

    @property
    def represented_by_phi(self) -> bool:
        return bool(self.phi_blocks)


def _typed_region_occurrence(
    value: SSAValue,
    canonical: SSAValue,
    authoritative: Meta | None,
) -> SSAValue:
    """Retype one region occurrence without erasing its authored view.

    A repository SSA id identifies storage, not one eternal tensor shape.
    Reshape/view operations intentionally reuse the id with different shapes.
    Dtype reconciliation may use the region-wide contract, but a non-empty
    occurrence shape is more specific and must survive.
    """

    dtype = (
        str(authoritative.dtype)
        if authoritative is not None
        and str(authoritative.dtype or "") not in {"", "unknown"}
        else value.dtype or canonical.dtype
    )
    shape = tuple(value.shape or canonical.shape or ())
    device = value.device or canonical.device
    if (
        value.dtype == dtype
        and tuple(value.shape or ()) == shape
        and value.device == device
    ):
        return value
    return replace(value, dtype=dtype, shape=shape, device=device)


@dataclass(frozen=True)
class PrecompileSSALoweringResult:
    module: IRModule
    validation: PrecompileSSAValidationResult
    shortfalls: tuple[SSALoweringShortfall, ...]
    cycles: tuple[SSACycle, ...]

    @property
    def complete(self) -> bool:
        return (
            self.validation.valid_precompile
            and not self.shortfalls
        )

    def shortfall_report(self) -> str:
        lines = []
        if self.validation.compatibility_shortfalls:
            lines.append(
                self.validation.compatibility_shortfall_report()
            )
        if self.shortfalls:
            lines.append("precompile-to-SSA lowering shortfalls:")
            lines.extend(
                f"- {item.domain}:{item.name} at {item.location}; "
                f"{item.reason}"
                for item in self.shortfalls
            )
        return "\n".join(lines) or "precompile-to-SSA lowering: complete"


def _ssa_value(value_id: int, meta: Meta | None) -> SSAValue:
    dtype = None if meta is None else meta.dtype
    if dtype is not None:
        dtype = str(dtype)
    if dtype is not None and "." in dtype:
        # Capturing through Torch records qualified spellings such as
        # ``torch.bool``. SSA and every compiled ABI use canonical dtype
        # names; allowing the frontend backend's module prefix to escape here
        # makes the Fortran declaration and C sidecar disagree.
        dtype = dtype.rsplit(".", 1)[-1]
    return SSAValue(
        int(value_id),
        dtype=dtype,
        shape=(
            ()
            if meta is None or meta.shape is None
            else tuple(meta.shape)
        ),
        device=None if meta is None else meta.device,
    )


def lower_fused_program_to_ssa(
    program: FusedProgram,
    *,
    function_name: str = "numerical_precompile",
) -> tuple[Function, tuple[SSALoweringShortfall, ...]]:
    """Translate only operations already named by the existing SSA registry.

    .. warning::

       **INTERNAL NUMERIC LOWERER -- DO NOT CALL THIS AS A COMPILER ENTRYPOINT.**

       No person-facing workflow and no orchestration, publishing, inspection,
       or application function may call this lowerer directly.  It accepts an
       already-precompiled :class:`FusedProgram`, not Python or a general AST.
       Only the precompiler and tensor-backend boundary may access it.  All
       other callers must enter through the appropriate precompiler or tensor
       backend and let that layer supply the numeric program.

       Tests may call this function only to verify the lowerer itself; a test
       call is not precedent for production use.
    """

    program = canonicalize_elementwise_steps(program)

    from .evolution_metagraph import (
        EvolutionComponentRef,
        active_evolution_metagraph,
        record_fused_program_evolution,
    )

    evolution = active_evolution_metagraph()
    precompile_evolution = record_fused_program_evolution(program)
    ssa_evolution = (
        None
        if evolution is None
        else evolution.open_graph("ssa", function_name)
    )

    metadata = program.meta or {}
    values = {
        int(value_id): _ssa_value(int(value_id), metadata.get(value_id))
        for value_id in (
            set(program.feeds)
            | {
                int(step.result_id)
                for step in program.steps
                if isinstance(step, OpStep)
            }
        )
    }
    arguments = [values[value_id] for value_id in sorted(program.feeds)]
    available = set(program.feeds)
    instructions: list[Instr] = []
    shortfalls: list[SSALoweringShortfall] = []

    if evolution is not None and ssa_evolution is not None:
        for value_id in sorted(program.feeds):
            sources = (
                (EvolutionComponentRef(precompile_evolution.id, str(value_id)),)
                if precompile_evolution is not None else ()
            )
            target = evolution.component(
                ssa_evolution,
                value_id,
                label=values[value_id].name(),
                kind="argument",
                consumes=sources,
            )
            if sources:
                evolution.handoff(
                    target,
                    sources,
                    transformation="precompile-to-ssa",
                )

    def element_count(value_id: int) -> int:
        meta = metadata.get(value_id)
        count = 1
        for extent in (() if meta is None else (meta.shape or ())):
            count *= int(extent)
        return count

    def tensor_algorithm(step: OpStep) -> str | None:
        if step.op_name == "random_source":
            return XOROSHIRO128SS_FILL
        translations = translations_for_operation(step.op_name)
        if not translations:
            return None
        symbols = {item.llvm_symbol for item in translations}
        if step.op_name == "slice":
            return (
                "index_select_double"
                if step.attrs.get("slice_kind") == "index_select"
                else "slice_copy_double"
            )
        if (
            "binary_scalar_double" in symbols
            and len(step.input_ids) == 1
            and any(name.endswith("_scalar") for name in step.attrs)
        ):
            return "binary_scalar_double"
        if {
            "binary_double",
            "binary_scalar_double",
        } <= symbols and len(step.input_ids) == 2:
            left_count, right_count = (
                element_count(value_id)
                for value_id in step.input_ids
            )
            return (
                "binary_scalar_double"
                if 1 in {left_count, right_count}
                and left_count != right_count
                else "binary_double"
            )
        if step.op_name == "sum":
            return (
                "reduce_dim_double"
                if step.attrs.get("axis") is not None
                else "sum_double"
            )
        # Prefer the public tensor kernel over scalar helpers when more than
        # one exact C/LLVM correspondence names the operation.
        for item in translations:
            if item.llvm_symbol != "binary_value":
                return item.llvm_symbol
        return translations[0].llvm_symbol

    for index, step in enumerate(program.steps):
        argument_count = step.attrs.get("argument_count")
        if (
            step.attrs.get("python_replacement_kind") == "operator"
            and isinstance(argument_count, int)
            and 0 <= argument_count < len(step.input_ids)
        ):
            # Intrinsic-call capture can retain the callable/static-reference
            # node before the authored operands.  The identity receipt states
            # the actual Python arity; only its trailing authored operands
            # belong to the numeric SSA call.  Passing the reference pointer
            # made unary ``float(x)`` a two-argument native cast and likewise
            # shifted _restore_type's cast_like value/reference pair.
            step = replace(
                step,
                input_ids=(
                    list(step.input_ids[-argument_count:])
                    if argument_count else []
                ),
            )
        location = f"{function_name}:step[{index}]"
        algorithm = tensor_algorithm(step)
        handler = (
            Handler.Const
            if step.op_name == "tensor_from_list"
            else Handler.Call
            if algorithm is not None
            else ssa_handler_for_precompile(step.op_name)
        )
        if handler is None:
            shortfalls.append(
                SSALoweringShortfall(
                    "numerical",
                    step.op_name,
                    location,
                    "operation has no name in the existing SSA Handler set",
                )
            )
            continue
        missing = tuple(
            value_id
            for value_id in step.input_ids
            if value_id not in available
        )
        if missing:
            shortfalls.append(
                SSALoweringShortfall(
                    "numerical",
                    step.op_name,
                    location,
                    "operands have no lowered SSA producer: "
                    + ", ".join(map(str, missing)),
                )
            )
            continue
        attributes = dict(step.attrs)
        if handler is Handler.Call:
            attributes.setdefault("tensor_operation", step.op_name)
        if algorithm is not None:
            attributes.update({
                "callee": algorithm,
                "tensor_operation": step.op_name,
                "lowered_from": "c_backend_llvm_ssa.TRANSLATIONS",
            })
        if step.op_name == "tensor_from_list":
            attributes.update({
                "value": attributes.get("data"),
                "tensor_operation": step.op_name,
            })
        instructions.append(
            Instr(
                handler.value,
                [values[value_id] for value_id in step.input_ids],
                values[step.result_id],
                attributes=attributes,
            )
        )
        if evolution is not None and ssa_evolution is not None:
            sources = (
                (EvolutionComponentRef(precompile_evolution.id, str(step.result_id)),)
                if precompile_evolution is not None else ()
            )
            target = evolution.component(
                ssa_evolution,
                step.result_id,
                label=str(handler.value),
                kind="instruction",
                attributes={"tensor_operation": step.op_name},
                consumes=sources,
            )
            if sources:
                evolution.handoff(
                    target,
                    sources,
                    transformation="precompile-to-ssa",
                )
            for position, value_id in enumerate(step.input_ids):
                input_ref = EvolutionComponentRef(
                    ssa_evolution.id,
                    str(value_id),
                )
                if evolution.has_component(input_ref):
                    evolution.relationship(
                        ssa_evolution,
                        input_ref,
                        target,
                        role=f"arg{position}",
                    )
        available.add(step.result_id)

    output_values = [
        values[value_id]
        for value_id in program.outputs.values()
        if value_id in available
    ]
    missing_outputs = tuple(
        (name, value_id)
        for name, value_id in program.outputs.items()
        if value_id not in available
    )
    for name, value_id in missing_outputs:
        shortfalls.append(
            SSALoweringShortfall(
                "numerical",
                "output",
                f"{function_name}:output[{name}]",
                f"value {value_id} has no lowered SSA producer",
            )
        )
    instructions.append(Instr(Handler.Ret.value, output_values, None))
    feed_origins = dict(
        (getattr(program, "extras", None) or {}).get(
            "capture_feed_origins", {}
        )
    )
    identity_tokens = dict(
        (getattr(program, "extras", None) or {}).get(
            "ssa_identity_tokens", {}
        )
    )
    parameter_names = tuple(
        (
            str(
                feed_origins.get(
                    value.id, feed_origins.get(str(value.id), {})
                ).get("binding_name", f"t{value.id}")
            ),
            int(value.id),
        )
        for value in arguments
    )
    function = Function(
        function_name,
        arguments,
        {
            "entry": BasicBlock(
                "entry",
                instructions,
                [],
            )
        },
        metadata={
            "named_outputs": tuple(
                (str(name), int(value_id))
                for name, value_id in program.outputs.items()
                if value_id in available
            ),
            "parameter_names": parameter_names,
            "ssa_identity_tokens": tuple(
                (
                    int(value_id),
                    tuple(map(str, token_chain)),
                )
                for value_id, token_chain in sorted(
                    identity_tokens.items(), key=lambda item: int(item[0])
                )
            ),
        },
    )
    if evolution is not None and ssa_evolution is not None:
        evolution.bind_artifact(function, ssa_evolution)
        evolution.close_graph(ssa_evolution)
    return function, tuple(shortfalls)


def lower_fused_integral_to_repository_ssa(
    program: FusedProgram,
    *,
    function_name: str,
) -> tuple[
    IRModule,
    dict[str, tuple[SSAValue, ...]],
    tuple[str, ...],
    tuple[SSALoweringShortfall, ...],
]:
    """Publish one planner-isolated numeric integral as repository SSA.

    This is the precompiler boundary for deterministic subdivision workers;
    orchestration does not call the numeric lowerer directly. The enclosing
    authored control owner remains source fallback until it can safely link a
    verified child deployment.
    """

    structural_contract = dict(
        (getattr(program, "extras", None) or {}).get(
            "structural_resident_table_contract"
        ) or {}
    )
    if structural_contract:
        sequences = tuple(map(
            dict, structural_contract.get("sequences") or (),
        ))
        stores = tuple(map(
            dict, structural_contract.get("stores") or (),
        ))
        declarations = tuple(
            (
                int(sequence["sequence_id"]),
                str(sequence["policy"]),
                int(sequence["column_count"]),
                bool(sequence["writable"]),
            )
            for sequence in sequences
        )
        column_dtypes = {
            int(sequence["sequence_id"]): tuple(map(
                str, sequence.get("column_dtypes") or (),
            ))
            for sequence in sequences
        }
        table_stores = tuple(
            (
                int(store["effect_value_id"]),
                int(store["key_value_id"]),
                int(store["stored_value_id"]),
                int(store["sequence_value_id"]),
            )
            for store in stores
        )
        value_meta = dict(program.meta or {})
        for sequence in sequences:
            sequence_id = int(sequence["sequence_id"])
            dtypes = column_dtypes[sequence_id]
            if dtypes:
                value_meta[sequence_id] = Meta((), dtypes[0])
        for store in stores:
            sequence = next(
                item for item in sequences
                if int(item["sequence_id"])
                == int(store["sequence_value_id"])
            )
            dtypes = column_dtypes[int(sequence["sequence_id"])]
            value_meta[int(store["key_value_id"])] = Meta((), dtypes[0])
            value_meta[int(store["stored_value_id"])] = Meta((), dtypes[-1])
        authored_ids = {
            *map(int, program.feeds),
            *(int(step.result_id) for step in program.steps),
            *(int(value_id) for value_id in program.outputs.values()),
        }
        first_value_id = max(authored_ids, default=-1) + 1
        control = ControlProgram(SequenceBlock(()))
        function, shortfalls = lower_control_program_to_ssa(
            control,
            function_name=str(function_name),
            first_value_id=first_value_id,
            region_value_meta=value_meta,
            sequence_declarations=declarations,
            sequence_column_dtypes=column_dtypes,
            source_sequence_ids=tuple(
                int(sequence["sequence_id"]) for sequence in sequences
            ),
            retained_sequence_ids=tuple(
                int(sequence["sequence_id"]) for sequence in sequences
            ),
            table_epilogue_operations=tuple(
                ("store", store) for store in table_stores
            ),
        )
        identity_tokens = dict(
            (getattr(program, "extras", None) or {}).get(
                "ssa_identity_tokens", {}
            )
        )
        function.metadata.update({
            "ssa_identity_tokens": tuple(
                (int(value_id), tuple(map(str, token_chain)))
                for value_id, token_chain in sorted(
                    identity_tokens.items(), key=lambda item: int(item[0])
                )
            ),
            "structural_integral_contract": structural_contract,
            "structural_integral_kind": "resident-table-mutation",
        })
        record_by_value = {
            int(store["stored_value_id"]): str(sequence["value_record"])
            for store in stores
            for sequence in sequences
            if int(sequence["sequence_id"])
            == int(store["sequence_value_id"])
            and sequence.get("value_record") is not None
        }
        for argument in function.args:
            record_identity = record_by_value.get(int(argument.id))
            if record_identity is not None:
                argument.accounting.update({
                    "structural_record_identity": record_identity,
                    "structural_record_handle": True,
                })
        helper_functions, sequence_tables = _sequence_artifacts_from_control(
            function
        )
        module = IRModule({
            str(function_name): function,
            **helper_functions,
        })
        module.sequence_tables.update(sequence_tables)
        return (
            module,
            {str(function_name): ()},
            (str(function_name),),
            tuple(shortfalls),
        )

    function, shortfalls = lower_fused_program_to_ssa(
        program, function_name=str(function_name),
    )
    returns = tuple(
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {"Ret", "ret", "Return", "return"}
    )
    output_values = tuple(returns[-1].args) if returns else ()
    return (
        IRModule({str(function_name): function}),
        {str(function_name): output_values},
        (str(function_name),),
        tuple(shortfalls),
    )


_LOG = logging.getLogger(__name__)


def _canonical_sequence_dtype(dtype: str | None) -> str | None:
    """Widen any sub-int64 integer dtype to int64.

    A sequence's storage array is one shared piece of memory that every
    function touching that sequence declares independently (each
    ``_ControlSSABuilder`` instance is per-function and infers column
    dtype from whatever local evidence it has -- a query operand's dtype,
    a fill literal's Python type). Two functions inferring different
    integer widths for the same shared array is a real element-address
    mismatch, not a cosmetic type error. Collapsing every integer
    candidate to one fixed width here, at the sequence descriptor's sole
    construction point, makes every independent inference agree by
    construction instead of requiring them to be reconciled after the
    fact.
    """
    if dtype in {"int", "int8", "int16", "int32", "i32"}:
        return "int64"
    return dtype


@dataclass(frozen=True)
class ResolvedSequenceSchema:
    """One sequence's structural shape, resolved once across every function
    that touches it, before any single function's local view can lock in a
    shape another function disagrees with.

    ``_ControlSSABuilder`` instances are built one per function/region, each
    from that one function's own local ``sequence_declarations``/
    ``table_lookups``/``table_stores`` evidence. A sequence's numeric id is a
    genuinely global identity (it traces back to one shared ProcessGraph
    node), so two functions touching the same sequence can each build a
    ``SSASequenceDescriptor`` with a different number of storage cells --
    not just a different dtype, but a structurally different shape (whether
    a status cell or live-flags cell exists at all). That is a real
    memory-layout bug, not a cosmetic one: it shows up as a caller passing
    one SSA value to two formal positions a callee expects to be distinct.

    This schema carries only structural facts, never concrete SSA value
    ids -- each function still mints its own ids from its own disjoint local
    id namespace. ``column_count``/``policy`` are the two fields with no
    sensible "union": every function that touches this sequence must agree
    on them exactly, or the sequence's two conflicting authored shapes ARE
    the bug, and lowering should say so rather than silently pick one.
    ``writable``/``retains_deleted_rows``/``nested_table`` are booleans a
    survey resolves as an OR across every function -- if any one function
    needs the extra storage cell, the one shared array needs it too.
    """

    column_count: int
    policy: str
    writable: bool
    retains_deleted_rows: bool
    nested_table: bool
    nested_value_dtype: str | None = None


def resolve_sequence_schemas(
    shells: Iterable[Mapping[str, Any]],
    *,
    location: str = "sequence-schema-survey",
) -> tuple[dict[int, ResolvedSequenceSchema], tuple[SSALoweringShortfall, ...]]:
    """Resolve one structural schema per sequence_id from every shell's raw
    evidence, before any shell's ``_ControlSSABuilder`` lowers and locks in
    a shape a different shell might disagree with.

    Each item in ``shells`` is a mapping carrying the same raw tuples
    ``_ControlSSABuilder`` itself consumes for one function/region:
    ``sequence_declarations``, ``sequence_initializations``,
    ``table_deletions``, ``retained_sequence_ids``, ``nested_sequence_ids``.
    Missing keys default to ``()``, so callers may pass only what they have.
    """
    column_counts: dict[int, int] = {}
    policies: dict[int, str] = {}
    origins: dict[int, str] = {}
    writable_union: dict[int, bool] = {}
    retains_union: dict[int, bool] = {}
    nested_union: dict[int, bool] = {}
    conflicted: set[int] = set()
    shortfalls: list[SSALoweringShortfall] = []

    def _observe(
        sequence_id: object,
        column_count: object,
        policy: object,
        writable: object,
        origin: object,
    ) -> None:
        sid = int(sequence_id)
        if sid in column_counts and (
            column_counts[sid] != int(column_count) or policies[sid] != str(policy)
        ):
            if sid not in conflicted:
                shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence", "resolved-schema-conflict", location,
                    f"sequence value {sid} was declared with "
                    f"{column_counts[sid]} column(s)/policy {policies[sid]!r} "
                    f"in {origins[sid]} and "
                    f"{int(column_count)} column(s)/policy {str(policy)!r} "
                    f"in {str(origin)} -- every function touching a shared "
                    "sequence must agree on its shape",
                ))
                conflicted.add(sid)
            return
        column_counts[sid] = int(column_count)
        policies[sid] = str(policy)
        origins[sid] = str(origin)
        writable_union[sid] = writable_union.get(sid, False) or bool(writable)

    for shell in shells:
        function_name = shell.get("function_name", "<unknown-function>")
        sequence_names = dict(shell.get("sequence_names") or {})
        deletion_ids = {
            int(sequence_id)
            for _effect_id, _key_id, sequence_id, _storage_identity
            in shell.get("table_deletions", ())
            if sequence_id is not None
        }
        deletion_ids.update(int(sid) for sid in shell.get("retained_sequence_ids", ()))
        deletion_ids.update(int(sid) for sid in shell.get("deletion_sequence_ids", ()))
        nested_ids = {int(sid) for sid in shell.get("nested_sequence_ids", ())}
        for sequence_id, policy, column_count, writable in shell.get(
            "sequence_declarations", ()
        ):
            names = tuple(sequence_names.get(int(sequence_id), ()))
            origin = f"{function_name}{names!r}"
            _observe(sequence_id, column_count, policy, writable, origin)
            sid = int(sequence_id)
            retains_union[sid] = retains_union.get(sid, False) or sid in deletion_ids
            nested_union[sid] = nested_union.get(sid, False) or sid in nested_ids
        for sequence_id, policy, column_count in shell.get(
            "sequence_initializations", ()
        ):
            names = tuple(sequence_names.get(int(sequence_id), ()))
            origin = f"{function_name}{names!r}"
            descriptor_policy = (
                "duplicates"
                if str(policy).startswith(("fill=", "literal_bytes="))
                else "unique"
                if str(policy).startswith("literal_table=")
                else str(policy)
            )
            _observe(
                sequence_id, column_count, descriptor_policy, True, origin,
            )
            sid = int(sequence_id)
            retains_union[sid] = retains_union.get(sid, False) or sid in deletion_ids
            nested_union[sid] = nested_union.get(sid, False) or sid in nested_ids

    resolved = {
        sid: ResolvedSequenceSchema(
            column_count=column_counts[sid],
            policy=policies[sid],
            writable=writable_union.get(sid, False),
            retains_deleted_rows=retains_union.get(sid, False),
            nested_table=nested_union.get(sid, False),
        )
        for sid in column_counts
        if sid not in conflicted
    }
    return resolved, tuple(shortfalls)


def _control_expression_mapping(
    expression: ControlExpression | None,
) -> dict[str, object] | None:
    if expression is None:
        return None
    return {
        "op": str(expression.op),
        "operands": [
            _control_expression_mapping(operand)
            for operand in expression.operands
        ],
        "value_id": (
            None if expression.value_id is None else int(expression.value_id)
        ),
        "literal": expression.literal,
    }


class _ControlSSABuilder:
    def __init__(
        self,
        program: ControlProgram,
        *,
        function_name: str,
        first_value_id: int,
        region_callees: dict[int, str] | None,
        region_signatures: dict[
            int, tuple[tuple[int, ...], tuple[int, ...]]
        ] | None,
        region_feed_meta: Mapping[int, tuple[Meta, ...]] | None = None,
        region_value_meta: Mapping[int, Meta] | None = None,
        plan_callsite_bindings: Mapping[
            int, tuple[tuple[int, ...], tuple[int, ...]]
        ] | None = None,
        value_aliases: Mapping[int, int] | None = None,
        inout_value_ids: tuple[int, ...] = (),
        output_value_ids: tuple[int, ...] = (),
        named_output_histories: Mapping[str, tuple[int, ...]] | None = None,
        value_name_histories: Mapping[str, tuple[int, ...]] | None = None,
        parameter_names: tuple[str, ...] = (),
        sequence_initializations: tuple[tuple[int, str, int], ...] = (),
        sequence_declarations: tuple[tuple[int, str, int, bool], ...] = (),
        sequence_column_dtypes: Mapping[int, tuple[str, ...]] | None = None,
        sequence_record_identities: Mapping[int, str] | None = None,
        source_sequence_ids: tuple[int, ...] = (),
        sequence_memberships: tuple[tuple[int, int, int, bool], ...] = (),
        table_lookups: tuple[tuple[int, int | tuple[int, ...], int], ...] = (),
        lexical_table_lookup_result_ids: tuple[int, ...] = (),
        table_lookup_defaults: dict[int, int | float] | None = None,
        table_stores: tuple[
            tuple[int, int | tuple[int, ...], int, int], ...
        ] = (),
        table_deletions: tuple[
            tuple[int, int | tuple[int, ...], int | None, str], ...
        ] = (),
        retained_sequence_ids: tuple[int, ...] = (),
        nested_sequence_ids: tuple[int, ...] = (),
        joined_sequence_ids: tuple[int, ...] = (),
        joined_singleton_values: Mapping[int, int] | None = None,
        nested_row_target_ids: tuple[int, ...] = (),
        selected_nested_sequence_ids: tuple[int, ...] = (),
        variant_projected_target_ids: tuple[int, ...] = (),
        region_array_feed_ids: Mapping[int, tuple[int, ...]] | None = None,
        nested_row_projections: tuple[tuple[int, int, int, str], ...] = (),
        sequence_length_values: Mapping[int, int] | None = None,
        table_region_operations: Mapping[int, tuple[tuple[str, tuple[Any, ...]], ...]] | None = None,
        table_region_post_operations: Mapping[int, tuple[tuple[str, tuple[Any, ...]], ...]] | None = None,
        table_epilogue_operations: tuple[tuple[str, tuple[Any, ...]], ...] = (),
        resolved_sequence_schemas: Mapping[int, ResolvedSequenceSchema] | None = None,
    ):
        from .evolution_metagraph import (
            active_evolution_metagraph,
            record_control_program_evolution,
        )

        self.program = program
        self.evolution = active_evolution_metagraph()
        self.control_evolution = record_control_program_evolution(program)
        self.ssa_evolution = (
            None
            if self.evolution is None
            else self.evolution.open_graph("ssa", function_name)
        )
        self._evolution_source = None
        self._evolution_instruction = 0
        self.region_value_meta = dict(region_value_meta or {})
        self.value_aliases = {
            int(alias): int(source)
            for alias, source in (value_aliases or {}).items()
        }
        self.inout_value_ids = set(map(int, inout_value_ids))
        self.output_value_ids = tuple(map(int, output_value_ids))
        self.named_output_histories = {
            str(name): tuple(map(int, history))
            for name, history in (named_output_histories or {}).items()
        }
        self.value_name_histories = {
            str(name): tuple(map(int, history))
            for name, history in (value_name_histories or {}).items()
        }
        self.parameter_names = tuple(map(str, parameter_names))
        self.function_name = function_name
        self.next_value_id = int(first_value_id)
        self.blocks: dict[str, BasicBlock] = {}
        self.block_counts: dict[str, int] = {}
        self.shortfalls: list[SSALoweringShortfall] = []
        self.region_callees = dict(region_callees or {})
        # A region with a recorded signature is schedulable by definition; its
        # callee name follows the one convention every consumer already uses
        # (autogenesis, backend_sources, the mandelbrot demo all look for
        # ``numerical_region_N``). Defaulting here means a caller supplying
        # signatures alone -- the unit-level entry point does -- gets calls
        # emitted rather than a silently empty program.
        declared_indices = (
            *(region_signatures or {}),
            *(getattr(program, "region_indices", ()) or ()),
        )
        for signature_index in declared_indices:
            self.region_callees.setdefault(
                int(signature_index), f"numerical_region_{int(signature_index)}"
            )
        self.region_signatures = dict(region_signatures or {})
        self.region_feed_meta = {
            int(region): tuple(metadata)
            for region, metadata in (region_feed_meta or {}).items()
        }
        # callsite_id -> (caller argument value ids, caller result value ids).
        self.plan_callsite_bindings = dict(plan_callsite_bindings or {})
        self.plan_callsite_by_result = {
            int(result_id): int(callsite_id)
            for callsite_id, (_argument_ids, result_ids)
            in self.plan_callsite_bindings.items()
            for result_id in result_ids
        }
        self.emitted_plan_callsites: set[int] = set()
        # While-latch condition regions may repeat a preheader region whose
        # output list also contains carried seed identities.  The header Phi
        # owns those identities throughout the loop; latch recomputation must
        # project colliding outputs to scratch instead of rebinding the Phi.
        self.preserved_region_output_ids: set[int] = set()
        self.arguments: list[SSAValue] = []
        self.external_values: dict[int, SSAValue] = {}
        self.declared_parameter_only_ids: set[int] = set()
        self.validation_contracts: list[dict[str, object]] = []
        self.table_lookup_defaults = dict(table_lookup_defaults or {})
        self.lexical_table_lookup_result_ids = frozenset(map(
            int, lexical_table_lookup_result_ids,
        ))
        self.sequence_descriptors: dict[int, SSASequenceDescriptor] = {}
        self.sequence_record_identities = {
            int(sequence_id): str(identity)
            for sequence_id, identity in (
                sequence_record_identities or {}
            ).items()
        }
        self.resolved_sequence_schemas: dict[int, ResolvedSequenceSchema] = {
            int(sequence_id): schema
            for sequence_id, schema in (resolved_sequence_schemas or {}).items()
        }
        self.sequence_storage_values: dict[int, tuple[SSAValue, ...]] = {}
        self.sequence_status_values: dict[int, SSAValue] = {}
        self.sequence_helper_functions: dict[str, Function] = {}
        # A locally authored ``list[bytes]`` has two physical views: the
        # logical outer row count and the flattened byte stream consumed by
        # ``b"".join``. This is the same composite ABI already used for a
        # parameter transformed through ``list(source)`` plus ``join``; the
        # companion is a compile-time storage artifact, not a Python object.
        self.joined_sequence_ids = set(map(int, joined_sequence_ids))
        self.joined_singleton_values = {
            int(sequence_id): int(value_id)
            for sequence_id, value_id in (
                joined_singleton_values or {}
            ).items()
        }
        self.joined_flat_sequence_ids: dict[int, int] = {}
        self.iterable_source_ids = {
            int(iterable_id)
            for iterable_id, _target_id, _induction
            in program.iterable_bindings
        }
        # Heterogeneous destructured rows are columnar at the repository ABI:
        # (resident source, projection) -> independently typed arena.  Column
        # zero retains the source identity; later columns receive fresh SSA
        # identities and share its extent.
        self.projected_row_columns: dict[tuple[int, int], SSAValue] = {}
        self.nested_row_target_ids = set(map(int, nested_row_target_ids))
        self.selected_nested_sequence_ids = set(map(
            int, selected_nested_sequence_ids
        ))
        self.variant_projected_target_ids = set(map(
            int, variant_projected_target_ids
        ))
        self.region_array_feed_ids = {
            int(region): set(map(int, value_ids))
            for region, value_ids in (region_array_feed_ids or {}).items()
        }
        self.nested_row_projections = tuple(
            (int(base), int(column), int(result), str(induction))
            for base, column, result, induction in nested_row_projections
        )
        self.variant_row_values: dict[int, SSAValue] = {}
        self.variant_handle_columns: dict[tuple[int, int], SSAValue] = {}
        # An iterable element (or one projected field of a destructured row)
        # may itself be a sequence/record.  Its outer column stores an integer
        # child handle; caller-owned flattened arenas and strides turn that
        # handle into the row base passed to the consuming region.  The key
        # retains the exact authored source relationship instead of relying on
        # the target's spelling or inferred Python class.
        self.nested_child_rows: dict[
            tuple[str, int, int],
            tuple[SSAValue, SSAValue, SSAValue, SSAValue],
        ] = {}
        self.nested_iterable_targets = {
            int(iterable_id): int(target_id)
            for iterable_id, target_id, _induction in program.iterable_bindings
            if int(target_id) in self.nested_row_target_ids
        }
        self.child_table_selections: dict[
            int, tuple[SSAChildTablePoolDescriptor, SSAValue]
        ] = {}
        self.table_region_operations = {
            int(region): tuple(operations)
            for region, operations in (table_region_operations or {}).items()
        }
        self.table_region_post_operations = {
            int(region): tuple(operations)
            for region, operations in (
                table_region_post_operations or {}
            ).items()
        }
        self.table_epilogue_operations = tuple(table_epilogue_operations)
        self.uniform_values: dict[str, SSAValue] = {}
        # Lexical control names (currently loop induction variables) are SSA
        # values too.  Keeping them here lets a StateMachineTick dispatch on
        # the surrounding loop without degrading that name to a symbolic
        # host-side load.
        self.local_control_values: dict[str, SSAValue] = {}
        self.loop_targets: list[tuple[BasicBlock, BasicBlock]] = []
        # Each active loop records the carried values visible on every direct
        # break edge.  A post-loop value is not always the header Phi: a break
        # can follow a carried update which has not traversed the latch yet.
        # The exit therefore needs its own edge-aware Phi.
        self.loop_exit_contexts: list[dict[str, Any]] = []
        self.deployment_records: list[dict[str, Any]] = [
            {
                "region_id": int(region.region_id),
                "kind": str(region.kind),
                "schedule": str(region.schedule),
                "schedule_preference": str(region.schedule_preference),
                "lane_count": len(region.lanes),
                "iteration_space": region.iteration_space,
                "carried_aliases": tuple(region.carried_aliases),
                "recursion_region_id": region.recursion_region_id,
                "origin": str(region.origin),
                "source_loop_node_id": region.source_loop_node_id,
                "scale": int(region.scale),
                "join": region.join,
                "declared_lanes": tuple(region.lanes),
            }
            for region in program.deployment_regions
        ]
        deployment_ids = [
            int(record["region_id"]) for record in self.deployment_records
        ]
        if len(deployment_ids) != len(set(deployment_ids)):
            raise ValueError("control deployment region IDs must be unique")
        self.next_deployment_id = max(deployment_ids, default=-1) + 1
        self.declared_region_memberships: dict[
            int, list[tuple[int, int]]
        ] = {}
        for record in self.deployment_records:
            for lane in record["declared_lanes"]:
                for region_index in lane.region_indices:
                    self.declared_region_memberships.setdefault(
                        int(region_index), []
                    ).append((int(record["region_id"]), int(lane.index)))
        self.active_deployments: list[tuple[int, int]] = []
        self.recursion_regions = {
            int(region.region_id): {
                "kind": str(region.kind),
                "lower_as": str(region.lower_as),
                "control_ir": bool(region.control_ir),
                "members": tuple(map(int, region.members)),
                "control_members": tuple(map(
                    int, region.control_members
                )),
                "incoming": tuple(region.incoming),
                "outgoing": tuple(region.outgoing),
                "feedback": tuple(region.feedback),
            }
            for region in program.recursion_regions
        }
        self.ssa_recursion_table: dict[int, dict[str, Any]] = {}
        for uniform in program.uniforms:
            value = SSAValue(
                int(uniform.value_id),
                dtype=str(uniform.dtype),
            )
            self.uniform_values[str(uniform.name)] = value
            self.external_values[int(uniform.value_id)] = value
            self.arguments.append(value)
        # A source parameter is part of the authored function ABI even when
        # its only consumer is a PlanCall that will be materialized after all
        # method/control functions have been lowered.  Waiting for an ordinary
        # region/control use drops call-only parameters, leaving the later
        # static linker with an empty callee signature and no caller value to
        # bind.  Preserve the initial identity (the input before any authored
        # reassignment); sequence/table lowering may refine that same identity
        # to its richer storage contract below.
        for name in self.parameter_names:
            history = self.value_name_histories.get(str(name), ())
            if not history:
                continue
            value_id = int(history[0])
            if value_id in self.external_values:
                continue
            value = self._value_from_meta(value_id)
            self.external_values[value_id] = value
            self.arguments.append(value)
            self.declared_parameter_only_ids.add(value_id)
        if self.external_values:
            self.next_value_id = max(
                self.next_value_id,
                max(self.external_values) + 1,
            )
        signature_ids = {
            value_id
            for feeds, outputs in self.region_signatures.values()
            for value_id in (*feeds, *outputs)
        }
        if signature_ids:
            self.next_value_id = max(
                self.next_value_id,
                max(signature_ids) + 1,
            )
        # Descriptor cells and other compiler-minted storage must live above
        # the complete authored/control identity domain.  Reserving only the
        # numerical region signatures allowed a fresh sequence-length cell to
        # take (for example) ID 212 before the later control-generated
        # sequence carrying authored ID 212 was declared.  Same integer then
        # meant two physical values and made linked ABIs irreconcilable.
        reserved_control_ids = {
            *map(int, control_dependency_value_ids(program)),
            *(
                int(sequence_id)
                for sequence_id, _policy, _columns, _writable
                in sequence_declarations
            ),
        }
        self.sequence_length_values = {
            int(result_id): int(sequence_id)
            for result_id, sequence_id in (
                sequence_length_values or {}
            ).items()
        }
        if reserved_control_ids:
            self.next_value_id = max(
                self.next_value_id, max(reserved_control_ids) + 1
            )
        # A specialized callee may receive a heterogeneous payload directly,
        # after its owning loop has already been split into the caller.  Such
        # a value still has two ABI columns: its authored scalar identity and
        # a separate row arena.  Projected loop bindings populate these later;
        # create only the unbound row columns here.
        projected_variant_targets = {
            int(target_id)
            for _iterable_id, target_id, _induction, _projection
            in program.projected_iterable_bindings
        }
        for value_id in sorted(
            self.variant_projected_target_ids - projected_variant_targets
        ):
            row = self.fresh_value(
                dtype=str(self._value_from_meta(value_id).dtype or "unknown")
            )
            row.accounting.update({
                "unbound_variant_source_id": int(value_id),
                "variant_column": "row",
            })
            self.arguments.append(row)
            self.variant_row_values[int(value_id)] = row
        # The loop target is the element contract for a resident iterable.
        # Empty Python aggregates carry no observation from which to infer an
        # element dtype and have historically defaulted to logical storage.
        # Bind the iterable's flat arena dtype to its exact target value before
        # lowering emits any indexed loads; shape remains the container shape.
        for iterable_id, target_id, _induction in program.iterable_bindings:
            target = self._value_from_meta(int(target_id))
            if target.dtype in {None, "", "unknown"}:
                continue
            iterable = self.external_value(int(iterable_id))
            iterable.dtype = target.dtype
            iterable.accounting = {
                **dict(iterable.accounting or {}),
                "iterable_element_dtype": str(target.dtype),
                "iterable_target_value_id": int(target_id),
            }
        self.current = self.new_block("entry")
        deletion_sequence_ids = {
            int(sequence_id)
            for _effect_id, _key_id, sequence_id, _storage_identity
            in table_deletions
            if sequence_id is not None
        }
        deletion_sequence_ids.update(map(int, retained_sequence_ids))
        nested_sequence_ids = set(map(int, nested_sequence_ids))
        nested_value_dtypes: dict[int, str] = {}
        lookup_sequence_by_result = {
            int(result_id): int(sequence_id)
            for result_id, _query_id, sequence_id in table_lookups
        }
        for iterable_id, target_id, _induction in program.iterable_bindings:
            sequence_id = lookup_sequence_by_result.get(int(iterable_id))
            target_meta = self.region_value_meta.get(int(target_id))
            if sequence_id is not None and target_meta is not None:
                nested_value_dtypes[sequence_id] = str(target_meta.dtype)
        for sequence_id, policy, column_count, writable in sequence_declarations:
            authored_column_dtypes = tuple(
                (sequence_column_dtypes or {}).get(int(sequence_id), ())
            )
            column_dtypes: list[str | None] = [
                authored_column_dtypes[column]
                if column < len(authored_column_dtypes) else None
                for column in range(int(column_count))
            ]
            for _result_id, query_id, lookup_sequence_id in table_lookups:
                if int(lookup_sequence_id) != int(sequence_id):
                    continue
                query_ids = query_id if isinstance(query_id, tuple) else (query_id,)
                for column, query_value_id in enumerate(query_ids):
                    meta = self.region_value_meta.get(int(query_value_id))
                    if meta is not None and column < len(column_dtypes):
                        column_dtypes[column] = str(meta.dtype)
            for _effect_id, key_id, value_id, store_sequence_id in table_stores:
                if int(store_sequence_id) != int(sequence_id):
                    continue
                key_ids = key_id if isinstance(key_id, tuple) else (key_id,)
                for column, key_value_id in enumerate(key_ids):
                    meta = self.region_value_meta.get(int(key_value_id))
                    if meta is not None and column < len(column_dtypes):
                        column_dtypes[column] = str(meta.dtype)
                value_meta = self.region_value_meta.get(int(value_id))
                if value_meta is not None and column_dtypes:
                    column_dtypes[-1] = str(value_meta.dtype)
            self._sequence_descriptor(
                int(sequence_id),
                policy=str(policy),
                writable=bool(writable),
                location=f"{function_name}.sequence_declaration",
                column_count=int(column_count),
                retains_deleted_rows=int(sequence_id) in deletion_sequence_ids,
                nested_table=int(sequence_id) in nested_sequence_ids,
                nested_value_dtype=nested_value_dtypes.get(int(sequence_id)),
                element_dtype=(
                    "int" if int(sequence_id) in self.joined_sequence_ids
                    else None
                ),
                column_dtypes=tuple(column_dtypes),
            )
        # Caller-provided arrays are the physical frame for both authored
        # inputs and compiler locals.  Their logical lifetimes are different:
        # retained source sequences arrive populated, while a writable local
        # (for example ``out = bytearray()``) is newly empty on every function
        # invocation even when its storage frame is reused by a linked caller.
        retained_residents = {
            *map(int, retained_sequence_ids),
            *map(int, source_sequence_ids),
        }
        for sequence_id, descriptor in sorted(self.sequence_descriptors.items()):
            if not descriptor.writable or int(sequence_id) in retained_residents:
                continue
            zero_index = self.constant_value(0)
            zero_length = self.constant_value(0)
            length_address = self.fresh_value(dtype="ptr")
            self.emit(
                Handler.GetElementPtr,
                [
                    self.sequence_storage_values[int(sequence_id)][
                        len(descriptor.column_value_ids)
                    ],
                    zero_index,
                ],
                length_address,
                attributes={"binding": "ssa_local_sequence_length"},
            )
            self.emit(
                Handler.Store, [zero_length, length_address],
                attributes={
                    "binding": "ssa_local_sequence_initialize",
                    "sequence_id": int(sequence_id),
                },
            )
        # Allocate deterministic flattened companions only after all authored
        # sequence descriptors exist. Their IDs come from the ordinary SSA
        # allocator in sorted authored-sequence order and therefore remain a
        # reproducible compile-complementary artifact.
        for sequence_id in sorted(self.joined_sequence_ids):
            if int(sequence_id) not in self.sequence_descriptors:
                continue
            flat_sequence_id = int(self.next_value_id)
            self.next_value_id += 1
            flat = self._sequence_descriptor(
                flat_sequence_id,
                policy="duplicates",
                writable=True,
                location=f"{function_name}.joined_sequence_flat_view",
                element_dtype="int",
            )
            if flat is None:
                continue
            self.joined_flat_sequence_ids[int(sequence_id)] = flat_sequence_id
            flat_length = self.sequence_storage_values[flat_sequence_id][
                len(flat.column_value_ids)
            ]
            flat_length_address = self.fresh_value(dtype="ptr")
            self.emit(
                Handler.GetElementPtr,
                [flat_length, self.constant_value(0)],
                flat_length_address,
                attributes={"binding": "ssa_joined_sequence_length"},
            )
            self.emit(
                Handler.Store,
                [self.constant_value(0), flat_length_address],
                attributes={"binding": "ssa_joined_sequence_initialize"},
            )
        for sequence_id, policy, column_count in sequence_initializations:
            descriptor_policy = (
                "duplicates"
                if str(policy).startswith(("fill=", "literal_bytes="))
                else "unique"
                if str(policy).startswith("literal_table=")
                else str(policy)
            )
            element_dtype = None
            if str(policy).startswith("fill=") and ";count=" in str(policy):
                fill_literal = ast.literal_eval(
                    str(policy).split(";", 1)[0].split("=", 1)[1]
                )
                element_dtype = (
                    "bool" if isinstance(fill_literal, bool)
                    else "int" if isinstance(fill_literal, int)
                    else "float64" if isinstance(fill_literal, float)
                    else "int64" if fill_literal is None
                    else None
                )
            elif str(policy).startswith("literal_bytes="):
                element_dtype = "int"
            descriptor = self._sequence_descriptor(
                int(sequence_id),
                policy=descriptor_policy,
                writable=True,
                location=f"{function_name}.sequence_initialization",
                column_count=int(column_count),
                retains_deleted_rows=int(sequence_id) in deletion_sequence_ids,
                nested_table=int(sequence_id) in nested_sequence_ids,
                nested_value_dtype=nested_value_dtypes.get(int(sequence_id)),
                element_dtype=element_dtype,
            )
            if descriptor is None:
                continue
            zero = self.constant_value(0)
            zero_index = self.constant_value(0)
            length_address = self.fresh_value(dtype="ptr")
            self.emit(
                Handler.GetElementPtr,
                [
                    self.sequence_storage_values[int(sequence_id)][
                        len(descriptor.column_value_ids)
                    ],
                    zero_index,
                ],
                length_address,
                attributes={"binding": "ssa_sequence_length"},
            )
            self.emit(
                Handler.Store,
                [zero, length_address],
                attributes={"binding": "ssa_sequence_initialize_length"},
            )
            if str(policy).startswith("literal_table="):
                rows = tuple(ast.literal_eval(
                    str(policy).split("=", 1)[1]
                ))
                if any(len(tuple(row)) != len(descriptor.column_value_ids)
                       for row in rows):
                    self.shortfalls.append(SSALoweringShortfall(
                        "ssa-sequence", "literal-table", self.function_name,
                        f"table {sequence_id} rows do not match its columns",
                    ))
                    continue
                for offset, row in enumerate(rows):
                    index = self.constant_value(int(offset))
                    for column, literal in enumerate(tuple(row)):
                        address = self.fresh_value(dtype="ptr")
                        self.emit(
                            Handler.GetElementPtr,
                            [
                                self.sequence_storage_values[
                                    int(sequence_id)
                                ][column],
                                index,
                            ],
                            address,
                            attributes={
                                "binding": "ssa_sequence_literal_table"
                            },
                        )
                        value = self.fresh_value(
                            dtype=descriptor.column_dtypes[column]
                        )
                        self.emit(
                            Handler.Const, [], value,
                            attributes={"value": literal},
                        )
                        self.emit(
                            Handler.Store, [value, address],
                            attributes={
                                "binding": "ssa_sequence_literal_table"
                            },
                        )
                self.emit(
                    Handler.Store,
                    [self.constant_value(len(rows)), length_address],
                    attributes={
                        "binding": "ssa_sequence_literal_table_length"
                    },
                )
            elif str(policy).startswith("literal_bytes="):
                payload = bytes.fromhex(str(policy).split("=", 1)[1])
                for offset, byte in enumerate(payload):
                    index = self.constant_value(int(offset))
                    address = self.fresh_value(dtype="ptr")
                    self.emit(
                        Handler.GetElementPtr,
                        [self.sequence_storage_values[int(sequence_id)][0], index],
                        address,
                        attributes={"binding": "ssa_sequence_literal_bytes"},
                    )
                    value = self.fresh_value(dtype="int")
                    self.emit(
                        Handler.Const,
                        [],
                        value,
                        attributes={"value": int(byte)},
                    )
                    self.emit(
                        Handler.Store,
                        [value, address],
                        attributes={"binding": "ssa_sequence_literal_bytes"},
                    )
                count = self.constant_value(len(payload))
                self.emit(
                    Handler.Store,
                    [count, length_address],
                    attributes={"binding": "ssa_sequence_literal_length"},
                )
        for result_id, query_id, sequence_id, negate in sequence_memberships:
            descriptor = self.sequence_descriptors.get(int(sequence_id))
            if descriptor is None or not descriptor.key_columns:
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence",
                    "contains",
                    f"{function_name}.sequence_membership[{result_id}]",
                    "membership requires a declared key-column table",
                ))
                continue
            from .ir_sequence_tables import lower_sequence_contains

            helper_name = f"ssa_sequence_{int(sequence_id)}_contains"
            lowering = lower_sequence_contains(
                descriptor,
                function_name=helper_name,
                first_value_id=self.next_value_id,
            )
            self._register_sequence_lowering(lowering)
            query = self.external_value(int(query_id))
            call_result = (
                self.fresh_value(dtype="bool")
                if negate else SSAValue(int(result_id), dtype="bool")
            )
            if not negate:
                self.next_value_id = max(self.next_value_id, int(result_id) + 1)
                self.external_values[int(result_id)] = call_result
            self.emit(
                Handler.Call,
                [*self.sequence_storage_values[int(sequence_id)], query],
                call_result,
                attributes={
                    "callee": helper_name,
                    "source_linked": True,
                    "ssa_sequence_operation": "contains",
                    "sequence_id": int(sequence_id),
                },
            )
            if negate:
                result = SSAValue(int(result_id), dtype="bool")
                self.next_value_id = max(self.next_value_id, int(result_id) + 1)
                self.external_values[int(result_id)] = result
                self.emit(Handler.LNot, [call_result], result)
        scheduled_table_operations = {
            (
                "lookup" if str(kind) == "lookup_capture" else str(kind),
                tuple(operation),
            )
            for operations in (
                *self.table_region_operations.values(),
                self.table_epilogue_operations,
            )
            for kind, operation in operations
        }
        for result_id, query_id, sequence_id in table_lookups:
            if int(result_id) in self.lexical_table_lookup_result_ids:
                continue
            if ("lookup", (result_id, query_id, sequence_id)) in scheduled_table_operations:
                continue
            # A whole-program shell graph shows every linked frame's nodes,
            # so a lookup can be recorded here whose result this function
            # never consumes -- it is a callee's operation.  Emitting it
            # anyway materialized its operands as id()-carrying arguments
            # (dead code by definition: real value ids are monotonic), which
            # displaced the frame's positional public-span correlation.  The
            # owner is the function whose regions consume the result.
            if int(result_id) not in self.region_value_meta:
                continue
            self._emit_table_lookup(result_id, query_id, sequence_id)
        for effect_id, key_id, value_id, sequence_id in table_stores:
            if ("store", (effect_id, key_id, value_id, sequence_id)) in scheduled_table_operations:
                continue
            self._emit_table_store(effect_id, key_id, value_id, sequence_id)
        for effect_id, key_id, sequence_id, storage_identity in table_deletions:
            if ("delete", (effect_id, key_id, sequence_id, storage_identity)) in scheduled_table_operations:
                continue
            self._emit_table_delete(effect_id, key_id, sequence_id, storage_identity)

    def _table_query_values(
        self, query_ids: int | tuple[int, ...]
    ) -> tuple[SSAValue, ...]:
        return tuple(
            self.external_value(int(value_id))
            for value_id in (
                query_ids if isinstance(query_ids, tuple) else (query_ids,)
            )
        )

    def _emit_table_lookup(
        self, result_id: int, query_id: int | tuple[int, ...], sequence_id: int
    ) -> None:
        descriptor = self.sequence_descriptors.get(int(sequence_id))
        if descriptor is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-table", "lookup", self.function_name,
                f"table {sequence_id} has no sequence descriptor",
            ))
            return
        from .ir_sequence_tables import lower_table_lookup

        default_literal = self.table_lookup_defaults.get(int(result_id))
        helper_name = (
            f"ssa_sequence_{int(sequence_id)}_lookup_or_default"
            if default_literal is not None
            else f"ssa_sequence_{int(sequence_id)}_lookup"
        )
        lowering = lower_table_lookup(
            descriptor,
            function_name=helper_name,
            default_parameter=default_literal is not None,
            first_value_id=self.next_value_id,
        )
        self._register_sequence_lowering(lowering)
        result = SSAValue(
            int(result_id),
            dtype=("int" if descriptor.child_table_pool is not None
                   else descriptor.column_dtypes[
                       next(
                           column
                           for column in range(len(descriptor.column_value_ids))
                           if column not in descriptor.key_columns
                       )
                   ]),
        )
        self.next_value_id = max(self.next_value_id, int(result_id) + 1)
        self.external_values[int(result_id)] = result
        default_operands: tuple[SSAValue, ...] = ()
        if default_literal is not None:
            value_columns = tuple(
                column
                for column in range(len(descriptor.column_value_ids))
                if column not in descriptor.key_columns
            )
            default_dtype = (
                descriptor.column_dtypes[value_columns[0]]
                if len(value_columns) == 1 else "unknown"
            )
            default_value = self.fresh_value(dtype=str(default_dtype))
            self.emit(
                Handler.Const, [], default_value,
                attributes={"value": default_literal},
            )
            default_operands = (default_value,)
        self.emit(
            Handler.Call,
            [
                *self.sequence_storage_values[int(sequence_id)],
                self.sequence_status_values[int(sequence_id)],
                *self._table_query_values(query_id),
                *default_operands,
            ],
            result,
            attributes={
                "callee": helper_name,
                "source_linked": True,
                "ssa_sequence_operation": "lookup",
                "sequence_id": int(sequence_id),
            },
        )
        if descriptor.child_table_pool is not None:
            self.child_table_selections[int(result_id)] = (
                descriptor.child_table_pool, result
            )
    def _emit_table_store(
        self, _effect_id: int, key_id: int | tuple[int, ...],
        value_id: int, sequence_id: int
    ) -> None:
        descriptor = self.sequence_descriptors.get(int(sequence_id))
        if descriptor is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-table", "store", self.function_name,
                f"table {sequence_id} has no sequence descriptor",
            ))
            return
        from .ir_sequence_tables import lower_table_store

        helper_name = f"ssa_sequence_{int(sequence_id)}_store"
        lowering = lower_table_store(
            descriptor,
            function_name=helper_name,
            first_value_id=self.next_value_id,
        )
        self._register_sequence_lowering(lowering)
        self.emit(
            Handler.Call,
            [
                *self.sequence_storage_values[int(sequence_id)],
                self.sequence_status_values[int(sequence_id)],
                *self._table_query_values(key_id),
                self.external_value(int(value_id)),
            ],
            None,
            attributes={
                "callee": helper_name,
                "source_linked": True,
                "ssa_sequence_operation": "table_store",
                "sequence_id": int(sequence_id),
            },
        )

    def _sequence_row_index(
        self, sequence_id: int, index_id: int, literal: int | None,
    ) -> SSAValue:
        if literal is None:
            return self.external_value(int(index_id), dtype="int64")
        if int(literal) >= 0:
            return self.constant_value(int(literal))
        descriptor = self.sequence_descriptors[int(sequence_id)]
        length_cell = self.sequence_storage_values[int(sequence_id)][
            len(descriptor.column_value_ids)
        ]
        length = self.fresh_value(dtype="int64")
        self.emit(
            Handler.Load, [length_cell], length,
            attributes={
                "binding": "ssa_sequence_row_length",
                "sequence_id": int(sequence_id),
            },
        )
        normalized = self.fresh_value(dtype="int64")
        self.emit(
            Handler.Add, [length, self.constant_value(int(literal))], normalized,
            attributes={
                "binding": "ssa_sequence_negative_index",
                "sequence_id": int(sequence_id),
            },
        )
        return normalized

    def _emit_sequence_row_load(
        self, result_id: int, sequence_id: int, index_id: int,
        index_literal: int | None, column: int, _row_value_id: int,
    ) -> None:
        descriptor = self.sequence_descriptors.get(int(sequence_id))
        if descriptor is None or not 0 <= int(column) < len(
            descriptor.column_value_ids
        ):
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence", "row-load", self.function_name,
                f"sequence {sequence_id} has no column {column}",
            ))
            return
        index = self._sequence_row_index(
            int(sequence_id), int(index_id), index_literal
        )
        source = self.external_value(
            int(descriptor.column_value_ids[int(column)]),
            dtype=str(descriptor.column_dtypes[int(column)]),
        )
        self.indexed_load(
            source, index, int(result_id),
            attributes={
                "binding": "ssa_sequence_row_load",
                "sequence_id": int(sequence_id),
                "column": int(column),
            },
        )

    def _emit_sequence_row_store(
        self, result_id: int, sequence_id: int, index_id: int,
        index_literal: int | None, row_value_ids: tuple[int, ...],
        _row_aggregate_id: int,
    ) -> None:
        descriptor = self.sequence_descriptors.get(int(sequence_id))
        if descriptor is None or len(row_value_ids) != len(
            descriptor.column_value_ids
        ):
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence", "row-store", self.function_name,
                f"sequence {sequence_id} row width is unresolved",
            ))
            return
        index = self._sequence_row_index(
            int(sequence_id), int(index_id), index_literal
        )
        for column, (column_id, value_id) in enumerate(zip(
            descriptor.column_value_ids, row_value_ids
        )):
            address = self.fresh_value(dtype="ptr")
            self.emit(
                Handler.GetElementPtr,
                [self.external_value(int(column_id)), index],
                address,
                attributes={
                    "binding": "ssa_sequence_row_store",
                    "sequence_id": int(sequence_id),
                    "column": int(column),
                },
            )
            self.emit(
                Handler.Store,
                [self.external_value(int(value_id)), address],
                attributes={
                    "binding": "ssa_sequence_row_store",
                    "sequence_id": int(sequence_id),
                    "column": int(column),
                },
            )
        self.external_values[int(result_id)] = self.external_value(
            int(sequence_id)
        )

    def _emit_table_delete(
        self, effect_id: int, key_id: int | tuple[int, ...],
        sequence_id: int | None, storage_identity: str
    ) -> None:
        if sequence_id is None:
            match = re.fullmatch(r"nested-table-value:(\d+)", storage_identity)
            selection = (
                self.child_table_selections.get(int(match.group(1)))
                if match is not None else None
            )
            if selection is not None:
                pool, handle = selection
                from .ir_sequence_tables import lower_child_table_delete

                helper_name = (
                    f"ssa_child_table_{int(match.group(1))}_delete"
                )
                lowering = lower_child_table_delete(
                    pool,
                    function_name=helper_name,
                    first_value_id=self.next_value_id,
                )
                self._register_sequence_lowering(lowering)
                pool_arguments = [
                    self.external_value(value_id)
                    for value_id in (
                        *pool.column_value_ids,
                        pool.length_value_id,
                        pool.capacity_value_id,
                        pool.row_stride_value_id,
                        pool.status_value_id,
                        pool.live_flags_value_id,
                    )
                ]
                self.emit(
                    Handler.Call,
                    [*pool_arguments, handle, *self._table_query_values(key_id)],
                    None,
                    attributes={
                        "callee": helper_name,
                        "source_linked": True,
                        "ssa_sequence_operation": "child_table_delete",
                        "child_handle_value_id": int(match.group(1)),
                    },
                )
                return
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-table", "delitem",
                f"{self.function_name}.table_delete[{effect_id}]",
                "nested table deletion requires a resident nested-table "
                f"descriptor: storage={storage_identity}, key_value_id={key_id}",
            ))
            return
        descriptor = self.sequence_descriptors.get(int(sequence_id))
        if descriptor is None or descriptor.live_flags_value_id is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-table", "delitem",
                f"{self.function_name}.table_delete[{effect_id}]",
                f"table {sequence_id} has no tombstone-enabled descriptor",
            ))
            return
        from .ir_sequence_tables import lower_table_delete

        first_live = isinstance(key_id, tuple) and not key_id
        if first_live:
            from .ir_sequence_tables import lower_table_delete_first
            helper_name = f"ssa_sequence_{int(sequence_id)}_delete_first"
            lowering = lower_table_delete_first(
                descriptor,
                function_name=helper_name,
                first_value_id=self.next_value_id,
            )
        else:
            helper_name = f"ssa_sequence_{int(sequence_id)}_delete"
            lowering = lower_table_delete(
                descriptor,
                function_name=helper_name,
                first_value_id=self.next_value_id,
            )
        self._register_sequence_lowering(lowering)
        self.emit(
            Handler.Call,
            [
                *self.sequence_storage_values[int(sequence_id)],
                self.sequence_status_values[int(sequence_id)],
                *(self._table_query_values(key_id) if not first_live else ()),
            ],
            None,
            attributes={
                "callee": helper_name,
                "source_linked": True,
                "ssa_sequence_operation": (
                    "table_delete_first" if first_live else "table_delete"
                ),
                "sequence_id": int(sequence_id),
            },
        )

    def _emit_sequence_fill(
        self, sequence_id: int, literal: bool | int | float | None, count_id: int
    ) -> None:
        descriptor = self.sequence_descriptors.get(int(sequence_id))
        if descriptor is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence", "fill", self.function_name,
                f"sequence {sequence_id} has no descriptor",
            ))
            return
        from .ir_sequence_tables import lower_sequence_fill

        helper_name = f"ssa_sequence_{int(sequence_id)}_fill"
        lowering = lower_sequence_fill(
            descriptor,
            function_name=helper_name,
            first_value_id=self.next_value_id,
        )
        self._register_sequence_lowering(lowering)
        value = self.fresh_value(dtype=str(descriptor.column_dtypes[0]))
        self.emit(Handler.Const, [], value, attributes={"value": literal})
        status = self.fresh_value(dtype="int")
        self.emit(
            Handler.Call,
            [
                *self.sequence_storage_values[int(sequence_id)],
                value,
                self.external_value(int(count_id), dtype="int"),
            ],
            status,
            attributes={
                "callee": helper_name,
                "source_linked": True,
                "ssa_sequence_operation": "fill",
                "sequence_id": int(sequence_id),
            },
        )
        self.emit(
            Handler.Store,
            [status, self._sequence_status_address(
                self.sequence_status_values[int(sequence_id)]
            )],
            attributes={"binding": "ssa_sequence_status"},
        )

    def _emit_sequence_append_fill(
        self, sequence_id: int, literal: bool | int | float | None,
        count_id: int,
    ) -> None:
        descriptor = self.sequence_descriptors.get(int(sequence_id))
        if descriptor is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence", "append_fill", self.function_name,
                f"sequence {sequence_id} has no descriptor",
            ))
            return
        from .ir_sequence_tables import lower_sequence_append_fill

        helper_name = f"ssa_sequence_{int(sequence_id)}_append_fill"
        lowering = lower_sequence_append_fill(
            descriptor,
            function_name=helper_name,
            first_value_id=self.next_value_id,
        )
        self._register_sequence_lowering(lowering)
        value = self.fresh_value(dtype=str(descriptor.column_dtypes[0]))
        self.emit(Handler.Const, [], value, attributes={"value": literal})
        status = self.fresh_value(dtype="int")
        self.emit(
            Handler.Call,
            [
                *self.sequence_storage_values[int(sequence_id)],
                value,
                self.external_value(int(count_id), dtype="int"),
            ],
            status,
            attributes={
                "callee": helper_name,
                "source_linked": True,
                "ssa_sequence_operation": "append_fill",
                "sequence_id": int(sequence_id),
            },
        )
        self.emit(
            Handler.Store,
            [status, self._sequence_status_address(
                self.sequence_status_values[int(sequence_id)]
            )],
            attributes={"binding": "ssa_sequence_status"},
        )

    def _emit_sequence_append_slice(
        self, destination_id: int, source_id: int,
        lower_id: int | None, upper_id: int | None,
    ) -> None:
        self.ensure_plan_callsite_result(
            int(source_id), location=f"{self.function_name}.append_slice"
        )
        destination = self.sequence_descriptors.get(int(destination_id))
        source = self.sequence_descriptors.get(int(source_id))
        if destination is None or source is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence", "append_slice", self.function_name,
                "slice append requires resident source and destination "
                f"descriptors: destination={destination_id}, source={source_id}",
            ))
            return
        from .ir_sequence_tables import lower_sequence_append_slice

        helper_name = (
            f"ssa_sequence_{int(destination_id)}_append_slice_"
            f"{int(source_id)}"
        )
        lowering = lower_sequence_append_slice(
            destination,
            source,
            function_name=helper_name,
            first_value_id=self.next_value_id,
        )
        if not lowering.complete:
            self.shortfalls.extend(
                SSALoweringShortfall(
                    "ssa-sequence", item.code.value, self.function_name,
                    item.reason,
                )
                for item in lowering.shortfalls
            )
            return
        self._register_sequence_lowering(lowering)
        storage = tuple({
            value.id: value
            for value in (
                *self.sequence_storage_values[int(destination_id)],
                *self.sequence_storage_values[int(source_id)],
            )
        }.values())
        # A whole-sequence extend (``resident += other``) has no authored
        # bounds: 0 and a beyond-any-length constant span the source, and
        # the helper's Python-slice clipping does the rest.
        if lower_id is None:
            lower_value = self.fresh_value(dtype="int")
            self.emit(
                Handler.Const, [], lower_value, attributes={"value": 0}
            )
        else:
            lower_value = self.external_value(int(lower_id), dtype="int")
        if upper_id is None:
            upper_value = self.fresh_value(dtype="int")
            self.emit(
                Handler.Const, [], upper_value,
                attributes={"value": 2 ** 31 - 1},
            )
        else:
            upper_value = self.external_value(int(upper_id), dtype="int")
        status = self.fresh_value(dtype="int")
        self.emit(
            Handler.Call,
            [
                *storage,
                lower_value,
                upper_value,
            ],
            status,
            attributes={
                "callee": helper_name,
                "source_linked": True,
                "ssa_sequence_operation": "append_slice",
                "sequence_id": int(destination_id),
                "source_sequence_id": int(source_id),
            },
        )
        self.emit(
            Handler.Store,
            [status, self._sequence_status_address(
                self.sequence_status_values[int(destination_id)]
            )],
            attributes={"binding": "ssa_sequence_status"},
        )

    def _emit_sequence_reset(self, sequence_id: int) -> None:
        """Start one ordinary sequence-expression result at logical length 0."""

        descriptor = self.sequence_descriptors.get(int(sequence_id))
        storage = self.sequence_storage_values.get(int(sequence_id))
        if descriptor is None or storage is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence", "reset", self.function_name,
                f"sequence {sequence_id} has no resident descriptor",
            ))
            return
        zero = self.fresh_value(dtype="int64")
        self.emit(Handler.Const, [], zero, attributes={"value": 0})
        length_address = storage[len(descriptor.column_value_ids)]
        self.emit(
            Handler.Store, [zero, length_address],
            attributes={
                "binding": "ssa_sequence_expression_reset",
                "sequence_id": int(sequence_id),
            },
        )

    def _emit_sequence_pack_bits(
        self, destination_id: int, source_id: int, width_id: int,
        width_literal: bool = False,
        callee_reference: int | None = None,
        plan_callsite_id: int | None = None,
    ) -> None:
        destination = self.sequence_descriptors.get(int(destination_id))
        source = self.sequence_descriptors.get(int(source_id))
        if destination is None or source is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence", "pack_bits", self.function_name,
                "bit packing requires resident source and destination "
                f"descriptors: destination={destination_id}, source={source_id}",
            ))
            return
        from .ir_sequence_tables import lower_sequence_pack_bits

        helper_name = (
            f"ssa_sequence_{int(destination_id)}_pack_bits_{int(source_id)}"
        )
        lowering = lower_sequence_pack_bits(
            destination,
            source,
            function_name=helper_name,
            first_value_id=self.next_value_id,
        )
        if not lowering.complete:
            self.shortfalls.extend(
                SSALoweringShortfall(
                    "ssa-sequence", item.code.value, self.function_name,
                    item.reason,
                ) for item in lowering.shortfalls
            )
            return
        self._register_sequence_lowering(lowering)
        storage = tuple({
            value.id: value
            for value in (
                *self.sequence_storage_values[int(destination_id)],
                *self.sequence_storage_values[int(source_id)],
            )
        }.values())
        status = self.fresh_value(dtype="int")
        if width_literal:
            width = self.fresh_value(dtype="int")
            self.emit(
                Handler.Const, [], width,
                attributes={"value": int(width_id)},
            )
        else:
            width = self.external_value(int(width_id), dtype="int")
        self.emit(
            Handler.Call,
            [*storage, width],
            status,
            attributes={
                "callee": helper_name,
                "source_linked": True,
                "ssa_sequence_operation": "pack_bits",
                "sequence_id": int(destination_id),
                "source_sequence_id": int(source_id),
                "decomposed_plan_call": callee_reference is not None,
                "callee_reference": callee_reference,
                "plan_callsite_id": plan_callsite_id,
            },
        )
        self.emit(
            Handler.Store,
            [status, self._sequence_status_address(
                self.sequence_status_values[int(destination_id)]
            )],
            attributes={"binding": "ssa_sequence_status"},
        )

    def _emit_sequence_prepend(
        self, sequence_id: int, value_id: int,
    ) -> None:
        descriptor = self.sequence_descriptors.get(int(sequence_id))
        if descriptor is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence", "prepend", self.function_name,
                f"sequence {sequence_id} has no descriptor",
            ))
            return
        from .ir_sequence_tables import lower_sequence_prepend

        helper_name = f"ssa_sequence_{int(sequence_id)}_prepend"
        lowering = lower_sequence_prepend(
            descriptor,
            function_name=helper_name,
            first_value_id=self.next_value_id,
        )
        self._register_sequence_lowering(lowering)
        status = self.fresh_value(dtype="int")
        self.emit(
            Handler.Call,
            [
                *self.sequence_storage_values[int(sequence_id)],
                self.external_value(int(value_id), dtype=str(
                    descriptor.column_dtypes[0]
                )),
            ],
            status,
            attributes={
                "callee": helper_name,
                "source_linked": True,
                "ssa_sequence_operation": "prepend",
                "sequence_id": int(sequence_id),
            },
        )
        self.emit(
            Handler.Store,
            [status, self._sequence_status_address(
                self.sequence_status_values[int(sequence_id)]
            )],
            attributes={"binding": "ssa_sequence_status"},
        )

    def _emit_sequence_prepend_packed_bytes(
        self, destination_id: int, source_id: int,
        prefix_id: int, byte_width: int,
        plan_callsite_id: int | None = None,
    ) -> None:
        destination = self.sequence_descriptors.get(int(destination_id))
        source = self.sequence_descriptors.get(int(source_id))
        if destination is None or source is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence", "prepend_packed_bytes", self.function_name,
                "packed prepend requires resident source and destination "
                f"descriptors: destination={destination_id}, source={source_id}",
            ))
            return
        from .ir_sequence_tables import lower_sequence_prepend_packed_bytes

        helper_name = (
            f"ssa_sequence_{int(destination_id)}_prepend_packed_"
            f"{int(source_id)}"
        )
        lowering = lower_sequence_prepend_packed_bytes(
            destination,
            source,
            function_name=helper_name,
            first_value_id=self.next_value_id,
        )
        self._register_sequence_lowering(lowering)
        storage = tuple({
            value.id: value
            for value in (
                *self.sequence_storage_values[int(destination_id)],
                *self.sequence_storage_values[int(source_id)],
            )
        }.values())
        width_value = self.fresh_value(dtype="int")
        self.emit(
            Handler.Const, [], width_value,
            attributes={"value": int(byte_width)},
        )
        status = self.fresh_value(dtype="int")
        self.emit(
            Handler.Call,
            [
                *storage,
                self.external_value(int(prefix_id), dtype=str(
                    destination.column_dtypes[0]
                )),
                width_value,
            ],
            status,
            attributes={
                "callee": helper_name,
                "source_linked": True,
                "ssa_sequence_operation": "prepend_packed_bytes",
                "sequence_id": int(destination_id),
                "source_sequence_id": int(source_id),
                "decomposed_plan_call": True,
                "plan_callsite_id": plan_callsite_id,
            },
        )
        self.emit(
            Handler.Store,
            [status, self._sequence_status_address(
                self.sequence_status_values[int(destination_id)]
            )],
            attributes={"binding": "ssa_sequence_status"},
        )

    def emit_table_region_operations(self, region_index: int) -> bool:
        operations = self.table_region_operations.get(int(region_index), ())
        if not operations:
            return False
        replaces_region = True
        for kind, arguments in operations:
            if kind == "lookup":
                self._emit_table_lookup(*arguments)
            elif kind == "lookup_capture":
                # The lookup replaces only the keyed source instruction.  Its
                # resident scalar result is an ordinary capture of the mixed
                # arithmetic region that follows.
                self._emit_table_lookup(*arguments)
                replaces_region = False
            elif kind == "row_load":
                self._emit_sequence_row_load(*arguments)
            elif kind == "row_load_capture":
                self._emit_sequence_row_load(*arguments)
                replaces_region = False
            elif kind == "row_store":
                self._emit_sequence_row_store(*arguments)
            elif kind == "store":
                self._emit_table_store(*arguments)
            elif kind == "delete":
                self._emit_table_delete(*arguments)
            elif kind == "fill":
                self._emit_sequence_fill(*arguments)
            elif kind == "append_fill":
                self._emit_sequence_append_fill(*arguments)
            elif kind == "append_slice":
                self._emit_sequence_append_slice(*arguments)
            elif kind == "reset":
                self._emit_sequence_reset(*arguments)
            elif kind == "append_scalar":
                destination_id, value_id = arguments
                self.lower_sequence_mutation(
                    ControlSequenceMutation(
                        sequence_value_id=int(destination_id),
                        operator="append",
                        argument_value_ids=(int(value_id),),
                        effect_node_id=int(value_id),
                        policy="duplicates",
                        argument_kind="scalar",
                    ),
                    path=f"{self.function_name}.structural_concat",
                )
            elif kind == "prepend":
                self._emit_sequence_prepend(*arguments)
            elif kind == "prepend_packed_bytes":
                self._emit_sequence_prepend_packed_bytes(*arguments)
            elif kind == "pack_bits":
                self._emit_sequence_pack_bits(*arguments)
            elif kind == "structural_consumed":
                # Its only result is consumed by a later structural helper.
                # The source region has no independent runtime effect.
                pass
            else:
                raise ValueError(f"unknown scheduled table operation {kind!r}")
        return replaces_region

    def _register_sequence_lowering(self, lowering) -> None:
        """Install helpers and reserve their complete program-wide ID range."""

        for function in lowering.functions:
            self.sequence_helper_functions[function.name] = function
            helper_ids = {
                value.id
                for value in function.args
            }
            helper_ids.update(
                instruction.res.id
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            )
            self.next_value_id = max(
                self.next_value_id,
                max(helper_ids, default=-1) + 1,
            )

    def fresh_value(
        self,
        *,
        dtype: str | None = None,
        shape: tuple[int, ...] = (),
    ) -> SSAValue:
        value = SSAValue(self.next_value_id, dtype=dtype, shape=shape)
        self.next_value_id += 1
        return value

    def external_value(
        self,
        value_id: int,
        *,
        dtype: str | None = None,
    ) -> SSAValue:
        value_id = int(value_id)
        sequence_id = self.sequence_length_values.get(value_id)
        if sequence_id is not None and value_id not in self.external_values:
            descriptor = self.sequence_descriptors.get(int(sequence_id))
            if descriptor is not None:
                length_cell = self.sequence_storage_values[int(sequence_id)][
                    len(descriptor.column_value_ids)
                ]
                value = SSAValue(value_id, dtype="int64")
                self.emit(
                    Handler.Load, [length_cell], value,
                    attributes={
                        "binding": "ssa_sequence_length_value",
                        "sequence_id": int(sequence_id),
                    },
                )
                self.external_values[value_id] = value
        seen: set[int] = set()
        while value_id in self.value_aliases and value_id not in seen:
            seen.add(value_id)
            value_id = int(self.value_aliases[value_id])
        self.declared_parameter_only_ids.discard(value_id)
        value = self.external_values.get(value_id)
        if value is None:
            value = self._value_from_meta(value_id, dtype=dtype)
            self.external_values[value_id] = value
            self.arguments.append(value)
        elif dtype is not None and str(value.dtype or "unknown") != str(dtype):
            # A structural control use can carry a stronger contract than an
            # earlier generic region capture.  Refine the one SSA identity
            # before the Function signature is assembled.
            refined = SSAValue(
                value.id,
                dtype=str(dtype),
                shape=tuple(value.shape),
                device=value.device,
                accounting=dict(value.accounting),
            )
            self.external_values[value_id] = refined
            self.arguments[:] = [
                refined if item is value else item for item in self.arguments
            ]
            value = refined
        return value

    def _value_from_meta(
        self, value_id: int, *, dtype: str | None = None
    ) -> SSAValue:
        """An SSA value carrying whatever the owning region already knows.

        The control program names values that regions define; only the region
        records their dtype and shape.  Building the value without that turns
        an array into a shapeless scalar, which every pointer-based backend
        tolerates and Fortran cannot express.
        """

        meta = self.region_value_meta.get(int(value_id))
        if meta is None:
            return SSAValue(int(value_id), dtype=dtype)
        value = _ssa_value(int(value_id), meta)
        if dtype is None:
            return value
        return SSAValue(
            value.id,
            dtype=str(dtype),
            shape=tuple(value.shape),
            device=value.device,
            accounting=dict(value.accounting),
        )

    def produced_value(
        self,
        value_id: int,
        *,
        dtype: str | None = None,
        claim_provisional_definition: bool = False,
    ) -> SSAValue:
        value_id = int(value_id)
        value = self.external_values.get(value_id)
        if value is not None:
            if value in self.arguments:
                if (
                    claim_provisional_definition
                    and value_id not in self.inout_value_ids
                ):
                    # Region/control construction is not lexical emission
                    # order.  A later-emitted coordinator load can therefore
                    # be the real definition of a value that an earlier-built
                    # region call provisionally exposed as a root argument.
                    # Keep the same SSAValue object so all already-built uses
                    # now refer to this definition, and retire only its
                    # provisional public-ABI role.  This is exact for loop
                    # projections: the coordinator owns the definition and
                    # the value is explicitly not in/out storage.
                    self.arguments.remove(value)
                    if dtype is not None:
                        value.dtype = str(dtype)
                    return value
                # A preallocated arena is commonly both the initial value
                # entering control and the destination published by a later
                # region.  SSA versions the write; it is not an identity
                # conflict.  The source value ID stays in accounting so the
                # public arena-address policy can rotate the two versions.
                value = self.fresh_value(
                    dtype=dtype or value.dtype,
                    shape=tuple(value.shape),
                )
                value.accounting.update({
                    **dict(self.external_values[value_id].accounting or {}),
                    "source_value_id": value_id,
                    **(
                        {"ssa_inout_write_version": True}
                        if value_id in self.inout_value_ids else {}
                    ),
                })
                self.external_values[value_id] = value
            return value
        value = self._value_from_meta(value_id, dtype=dtype)
        self.external_values[value_id] = value
        return value

    def constant_value(self, literal: int) -> SSAValue:
        value = self.fresh_value(dtype="int")
        self.emit(
            Handler.Const,
            [],
            value,
            attributes={"value": int(literal)},
        )
        return value

    def indexed_load(
        self,
        source: SSAValue,
        index: SSAValue | tuple[SSAValue, ...] | list[SSAValue],
        result_id: int,
        *,
        attributes: dict[str, Any],
        claim_provisional_definition: bool = False,
    ) -> SSAValue:
        address = self.fresh_value(dtype="ptr")
        indices = (
            tuple(index)
            if isinstance(index, (tuple, list))
            else (index,)
        )
        self.emit(
            Handler.GetElementPtr,
            [source, *indices],
            address,
            attributes=attributes,
        )
        result = self.produced_value(
            result_id,
            dtype=str(source.dtype or "unknown"),
            claim_provisional_definition=claim_provisional_definition,
        )
        self.emit(Handler.Load, [address], result, attributes=attributes)
        return result

    def bind_nested_row(
        self,
        source: SSAValue,
        induction: SSAValue,
        target_id: int,
        *,
        child_key: tuple[str, int, int],
        attributes: dict[str, Any],
    ) -> SSAValue:
        """Bind one caller-owned child row selected by an integer handle.

        Nested collections cross the repository ABI as two ordinary tables:
        the outer column contains integer row handles and a flattened child
        arena contains the rows.  This emits the complete address calculation
        into SSA; neither the backend nor a runtime language reconstructs an
        object for the selected row.
        """

        selected_sequence = int(child_key[1]) in (
            self.selected_nested_sequence_ids
        )
        if selected_sequence:
            handle = source
        else:
            handle_address = self.fresh_value(dtype="ptr")
            handle = self.fresh_value(dtype="int")
            handle_attributes = {
                **attributes, "binding": "nested_row_handle"
            }
            self.emit(
                Handler.GetElementPtr,
                [source, induction],
                handle_address,
                attributes=handle_attributes,
            )
            self.emit(
                Handler.Load,
                [handle_address],
                handle,
                attributes=handle_attributes,
            )
        child = self._nested_child_storage(child_key, target_id)
        child_data, _child_lengths, sequence_stride, row_stride = child
        row_index = induction
        if selected_sequence:
            sequence_offset = self.fresh_value(dtype="int")
            row_index = self.fresh_value(dtype="int")
            self.emit(
                Handler.Mul,
                [handle, sequence_stride],
                sequence_offset,
                attributes={
                    **attributes, "binding": "nested_sequence_offset"
                },
            )
            self.emit(
                Handler.Add,
                [sequence_offset, induction],
                row_index,
                attributes={
                    **attributes, "binding": "nested_sequence_row"
                },
            )
        offset = self.fresh_value(dtype="int")
        row_base = self.fresh_value(dtype=str(child_data.dtype or "unknown"))
        row_base.accounting["source_value_id"] = int(target_id)
        self.emit(
            Handler.Mul,
            [row_index if selected_sequence else handle, row_stride],
            offset,
            attributes={**attributes, "binding": "nested_row_offset"},
        )
        self.emit(
            Handler.GetElementPtr,
            [child_data, offset],
            row_base,
            attributes={**attributes, "binding": "nested_row_base"},
        )
        self.external_values[int(target_id)] = row_base
        return row_base

    def _nested_child_storage(
        self,
        child_key: tuple[str, int, int],
        target_id: int,
    ) -> tuple[SSAValue, SSAValue, SSAValue, SSAValue]:
        child = self.nested_child_rows.get(child_key)
        if child is None:
            target_meta = self.region_value_meta.get(int(target_id))
            child_data = self.fresh_value(dtype=(
                None if target_meta is None else str(target_meta.dtype)
            ))
            child_lengths = self.fresh_value(dtype="int")
            sequence_stride = self.fresh_value(dtype="int")
            row_stride = self.fresh_value(dtype="int")
            child_data.accounting.update({
                "nested_row_source_kind": str(child_key[0]),
                "nested_row_source_id": int(child_key[1]),
                "nested_row_projection": int(child_key[2]),
            })
            self.arguments.extend((
                child_data, child_lengths, sequence_stride, row_stride
            ))
            child = (
                child_data, child_lengths, sequence_stride, row_stride
            )
            self.nested_child_rows[child_key] = child
        return child

    def emit_plan_callsite(self, callsite_id: int, *, location: str) -> None:
        """Lower a scheduled call statement to a placeholder Call.

        Arguments resolve through ``external_value`` at THIS position, which
        is the point of scheduling the call as a statement: inside a loop
        body the carried machinery maps them to the current iteration's
        values, instead of the post-hoc anchor insertion binding whatever id
        the graph happened to record. The placeholder carries the callsite id
        so frame linking can complete callee symbol and bindings in place --
        position from the plan, bindings from the linker.
        """

        bindings = self.plan_callsite_bindings.get(int(callsite_id))
        if bindings is None:
            self.shortfalls.append(SSALoweringShortfall(
                "control", "plan_callsite", location,
                f"scheduled callsite {callsite_id} has no recorded bindings",
            ))
            return
        argument_ids, result_ids = bindings
        arguments = [
            self.external_value(int(value_id)) for value_id in argument_ids
        ]
        marker_attributes = {
            "callee": f"__plan_callsite_{int(callsite_id)}__",
            "plan_callsite_marker": True,
            "plan_callsite_id": int(callsite_id),
            "output_ids": tuple(int(v) for v in result_ids),
        }
        if len(result_ids) > 1:
            # A multi-result call publishes EVERY bound caller result here,
            # at its scheduled position, through the same aggregate
            # convention a region call uses.  Publishing only the first
            # result left the others without a producer at this position;
            # inside a loop body that is exactly the carried update whose
            # latch operand the loop machinery must see produced.  The
            # placeholder projections own the result objects by identity;
            # frame linking rebinds the linked call's projections onto them.
            aggregate = self.fresh_value(dtype="ssa.aggregate")
            self.emit(
                Handler.Call,
                arguments,
                aggregate,
                attributes={
                    **marker_attributes,
                    "result_convention": "ssa.aggregate",
                },
            )
            for output_index, output_id in enumerate(result_ids):
                projection_attributes = {
                    "plan_callsite_marker_projection": True,
                    "plan_callsite_id": int(callsite_id),
                    "aggregate_index": int(output_index),
                    "source_output_id": int(output_id),
                }
                address = self.fresh_value(dtype="ptr")
                self.emit(
                    Handler.GetElementPtr,
                    [aggregate, self.constant_value(output_index)],
                    address,
                    attributes=projection_attributes,
                )
                # The projection's dtype/shape are the region-declared facts
                # of the caller result, not the aggregate handle's dtype.
                result = self.produced_value(int(output_id))
                self.emit(
                    Handler.Load, [address], result,
                    attributes=projection_attributes,
                )
            self.emitted_plan_callsites.add(int(callsite_id))
            return
        result = None
        if result_ids:
            primary = int(result_ids[0])
            result = self.external_values.get(primary)
            if result is None:
                result = self._value_from_meta(primary)
                self.external_values[primary] = result
        self.emit(
            Handler.Call,
            arguments,
            result,
            attributes=marker_attributes,
        )
        self.emitted_plan_callsites.add(int(callsite_id))

    def emit_region_call(self, region_index: int, *, location: str) -> None:
        if self.emit_table_region_operations(region_index):
            return
        if region_index not in self.region_callees:
            # A scheduled region the lowering cannot call is a hole in the
            # program, not a no-op. Silently returning here produced exactly
            # the failure shape this tree keeps paying for -- a plan that
            # says N regions and an emitted function containing none of
            # them, with nothing raised.
            self.shortfalls.append(SSALoweringShortfall(
                "control", "region_callee", location,
                f"scheduled region {region_index} has no callee and no "
                "signature to default one from; the region's work is absent "
                "from the emitted function",
            ))
            return
        callee = self.region_callees[region_index]
        feeds, outputs = self.region_signatures.get(
            region_index, ((), ())
        )
        array_feeds = self.region_array_feed_ids.get(int(region_index), set())
        storage_arguments = [
            self.variant_row_values.get(int(value_id), self.external_value(value_id))
            if int(value_id) in array_feeds
            else self.external_value(value_id)
            for value_id in feeds
        ]
        feed_meta = self.region_feed_meta.get(int(region_index), ())
        arguments = [
            SSAValue(
                storage.id,
                dtype=(
                    str(feed_meta[position].dtype)
                    if position < len(feed_meta) else storage.dtype
                ),
                shape=(
                    tuple(feed_meta[position].shape)
                    if position < len(feed_meta) else tuple(storage.shape)
                ),
                device=(
                    feed_meta[position].device
                    if position < len(feed_meta) else storage.device
                ),
                accounting={
                    **dict(storage.accounting or {}),
                    "ssa_storage_alias": int(storage.id),
                    "ssa_region_feed": (int(region_index), int(position)),
                    **(
                        {"ssa_loop_carried_feed": int(value_id)}
                        if int(value_id) in self.preserved_region_output_ids
                        else {}
                    ),
                },
            )
            for position, (value_id, storage) in enumerate(zip(
                feeds, storage_arguments,
            ))
        ]
        aggregate = (
            self.fresh_value(dtype="ssa.aggregate")
            if outputs
            else None
        )
        self.emit(
            Handler.Call,
            arguments,
            aggregate,
            attributes={
                "callee": callee,
                "region_index": region_index,
                "feed_ids": feeds,
                "feed_shapes": tuple(tuple(value.shape) for value in arguments),
                "feed_dtypes": tuple(str(value.dtype or "") for value in arguments),
                "output_ids": outputs,
                "result_convention": "ssa.aggregate",
            },
        )
        if aggregate is not None:
            for output_index, output_id in enumerate(outputs):
                index = self.constant_value(output_index)
                attributes = {
                    "region_index": region_index,
                    "aggregate_index": output_index,
                    "source_output_id": output_id,
                }
                if int(output_id) in self.preserved_region_output_ids:
                    address = self.fresh_value(dtype="ptr")
                    self.emit(
                        Handler.GetElementPtr,
                        [aggregate, index],
                        address,
                        attributes=attributes,
                    )
                    contract = self._value_from_meta(int(output_id))
                    scratch = self.fresh_value(
                        dtype=contract.dtype,
                        shape=contract.shape,
                    )
                    self.emit(
                        Handler.Load, [address], scratch,
                        attributes={
                            **attributes,
                            "discarded_carried_seed_projection": True,
                        },
                    )
                else:
                    self.indexed_load(
                        aggregate,
                        index,
                        output_id,
                        attributes=attributes,
                        # Control construction can encounter a consumer before
                        # lexical scheduling emits the region which defines it.
                        # ``external_value`` provisionally exposes that identity as
                        # a formal.  This load is its exact planned definition, so
                        # claim the same SSAValue object and retire the provisional
                        # formal.  ``produced_value`` still refuses this transition
                        # for declared in/out storage and versions those writes.
                        claim_provisional_definition=True,
                    )
        for kind, table_arguments in self.table_region_post_operations.get(
            int(region_index), ()
        ):
            if kind == "lookup":
                self._emit_table_lookup(*table_arguments)
            elif kind == "store":
                self._emit_table_store(*table_arguments)
            elif kind == "delete":
                self._emit_table_delete(*table_arguments)
            elif kind == "fill":
                self._emit_sequence_fill(*table_arguments)
            elif kind == "append_fill":
                self._emit_sequence_append_fill(*table_arguments)
            elif kind == "append_slice":
                self._emit_sequence_append_slice(*table_arguments)
            elif kind == "prepend":
                self._emit_sequence_prepend(*table_arguments)
            elif kind == "prepend_packed_bytes":
                self._emit_sequence_prepend_packed_bytes(*table_arguments)
            elif kind == "pack_bits":
                self._emit_sequence_pack_bits(*table_arguments)
            elif kind == "extend":
                effect_id, destination_id, source_id = table_arguments
                self.lower_sequence_mutation(
                    ControlSequenceMutation(
                        sequence_value_id=int(destination_id),
                        operator="extend",
                        argument_value_ids=(int(source_id),),
                        effect_node_id=int(effect_id),
                        policy="duplicates",
                        argument_kind="sequence",
                    ),
                    path=f"{location}.structural",
                )
            else:
                raise ValueError(
                    f"unknown post-region table operation {kind!r}"
                )

    def new_block(self, stem: str) -> BasicBlock:
        count = self.block_counts.get(stem, 0)
        self.block_counts[stem] = count + 1
        name = stem if count == 0 else f"{stem}.{count}"
        block = BasicBlock(name)
        self.blocks[name] = block
        return block

    def _value_dominates_current_edge(self, value: SSAValue) -> bool:
        """Whether this exact SSA object is resident on the current edge."""

        if any(argument is value for argument in self.arguments):
            return True
        producer_blocks = {
            name
            for name, block in self.blocks.items()
            if any(instruction.res is value for instruction in block.instrs)
        }
        if not producer_blocks:
            return False
        current_name = self.current.name
        if current_name in producer_blocks:
            return True
        block_names = tuple(self.blocks)
        entry_name = block_names[0]
        predecessors = {name: set() for name in block_names}
        for name, block in self.blocks.items():
            for successor in block.successors:
                if successor in predecessors:
                    predecessors[successor].add(name)
        dominators = {
            name: ({name} if name == entry_name else set(block_names))
            for name in block_names
        }
        changed = True
        while changed:
            changed = False
            for name in block_names:
                if name == entry_name:
                    continue
                incoming = predecessors[name]
                common = (
                    set.intersection(*(
                        dominators[parent] for parent in incoming
                    ))
                    if incoming else set()
                )
                updated = {name} | common
                if updated != dominators[name]:
                    dominators[name] = updated
                    changed = True
        return len(producer_blocks) == 1 and next(iter(
            producer_blocks
        )) in dominators[current_name]

    def emit(
        self,
        op: Handler,
        args: list[SSAValue],
        result: SSAValue | None = None,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        resolved_attributes = dict(attributes or {})
        if self.active_deployments:
            resolved_attributes.setdefault(
                "deployment_memberships",
                tuple(self.active_deployments),
            )
        self.current.instrs.append(
            Instr(
                op.value,
                args,
                result,
                attributes=resolved_attributes,
            )
        )
        if self.evolution is not None and self.ssa_evolution is not None:
            from .evolution_metagraph import EvolutionComponentRef

            local_id = (
                str(result.id)
                if result is not None
                else f"{self.current.name}:{self._evolution_instruction}:{op.value}"
            )
            self._evolution_instruction += 1
            sources = (
                (self._evolution_source,)
                if self._evolution_source is not None else ()
            )
            target = self.evolution.component(
                self.ssa_evolution,
                local_id,
                label=str(op.value),
                kind="control-instruction",
                attributes={
                    "block": self.current.name,
                    "control_op": str(op.value),
                    **{
                        key: resolved_attributes[key]
                        for key in (
                            "deployment_memberships",
                            "deployment_frame",
                            "region_id",
                            "region_index",
                            "scale",
                            "join_mode",
                            "reduction_operator",
                            "allow_reassociation",
                            "schedule_preference",
                            "predicate_value_id",
                            "recursion_region_id",
                        )
                        if key in resolved_attributes
                    },
                },
                consumes=sources,
            )
            if sources:
                self.evolution.handoff(
                    target,
                    sources,
                    transformation="control-ir-to-ssa",
                )
            for position, argument in enumerate(args):
                argument_ref = EvolutionComponentRef(
                    self.ssa_evolution.id,
                    str(argument.id),
                )
                if self.evolution.has_component(argument_ref):
                    self.evolution.relationship(
                        self.ssa_evolution,
                        argument_ref,
                        target,
                        role=f"arg{position}",
                    )

    def emit_deployment_boundary(
        self, op: Handler, record: dict[str, Any]
    ) -> tuple[str, int]:
        """Emit a structural frame marker; it performs no numeric work."""
        site = (self.current.name, len(self.current.instrs))
        join = record["join"]
        self.emit(op, [], attributes={
            "deployment_frame": True,
            "region_id": int(record["region_id"]),
            "scale": int(record.get("scale", 1)),
            "join_mode": join.mode.value,
            "reduction_operator": join.reduction_operator,
            "allow_reassociation": bool(join.allow_reassociation),
            "schedule_preference": str(record["schedule_preference"]),
        })
        record["deploy_site" if op is Handler.Deploy else "join_site"] = site
        return site

    def branch(self, target: BasicBlock) -> None:
        self.emit(
            Handler.Br,
            [],
            attributes={"target": target.name},
        )
        self.current.successors.append(target.name)

    def conditional_branch(
        self,
        condition: SSAValue,
        if_true: BasicBlock,
        if_false: BasicBlock,
    ) -> None:
        self.emit(
            Handler.CondBr,
            [condition],
            attributes={
                "true_target": if_true.name,
                "false_target": if_false.name,
            },
        )
        self.current.successors.extend((if_true.name, if_false.name))

    def expression_value(
        self,
        expression: str,
        *,
        location: str,
    ) -> SSAValue:
        spelling = str(expression).strip()
        iterable_extent = re.fullmatch(
            r"__iterable_extent_(\d+)__", spelling
        )
        if iterable_extent is not None:
            iterable_id = int(iterable_extent.group(1))
            resident_id = iterable_id
            seen_aliases: set[int] = set()
            while (
                resident_id in self.value_aliases
                and resident_id not in seen_aliases
            ):
                seen_aliases.add(resident_id)
                resident_id = int(self.value_aliases[resident_id])
            resident = self.sequence_descriptors.get(resident_id)
            if resident is not None:
                length_cell = self.sequence_storage_values[resident_id][
                    len(resident.column_value_ids)
                ]
                extent = self.fresh_value(dtype="int64")
                self.emit(
                    Handler.Load, [length_cell], extent,
                    attributes={
                        "binding": "resident_iterable_length",
                        "sequence_id": int(resident_id),
                        "source_value_id": iterable_id,
                    },
                )
                return extent
            child_selection = self.child_table_selections.get(iterable_id)
            if child_selection is not None:
                pool, handle = child_selection
                length_address = self.fresh_value(dtype="ptr")
                extent = self.fresh_value(dtype="int")
                self.emit(
                    Handler.GetElementPtr,
                    [self.external_value(pool.length_value_id), handle],
                    length_address,
                    attributes={"binding": "child_table_length"},
                )
                self.emit(
                    Handler.Load, [length_address], extent,
                    attributes={"binding": "child_table_length"},
                )
                return extent
            nested_target_id = self.nested_iterable_targets.get(iterable_id)
            if nested_target_id is not None:
                child_key = ("iterable", iterable_id, -1)
                _child_data, child_lengths, _sequence_stride, _row_stride = (
                    self._nested_child_storage(child_key, nested_target_id)
                )
                handle = self.external_value(iterable_id, dtype="int")
                length_address = self.fresh_value(dtype="ptr")
                extent = self.fresh_value(dtype="int")
                self.emit(
                    Handler.GetElementPtr,
                    [child_lengths, handle],
                    length_address,
                    attributes={"binding": "nested_row_length"},
                )
                self.emit(
                    Handler.Load, [length_address], extent,
                    attributes={"binding": "nested_row_length"},
                )
                return extent
            source = self.external_value(iterable_id)
            extent = self.fresh_value(dtype="int")
            self.emit(
                Handler.Call,
                [source],
                extent,
                attributes={
                    "tensor_operation": "extent",
                    "extent_kind": "dim",
                    "axis": 0,
                    "binding": "iterable_extent",
                    "source_value_id": iterable_id,
                },
            )
            return extent
        value_match = re.fullmatch(r"value_(\d+)", spelling)
        if value_match is not None:
            return self.external_value(int(value_match.group(1)))
        uniform = self.uniform_values.get(spelling)
        if uniform is not None:
            return uniform
        local = self.local_control_values.get(spelling)
        if local is not None:
            return local
        try:
            literal = int(spelling, 10)
        except ValueError:
            lowered = self._control_arithmetic_value(spelling)
            if lowered is not None:
                return lowered
            value = self.fresh_value(dtype="int")
            self.emit(
                Handler.Load,
                [],
                value,
                attributes={"control_expression": spelling},
            )
            self.shortfalls.append(
                SSALoweringShortfall(
                    "control",
                    "expression",
                    location,
                    f"control expression {spelling!r} is retained as a "
                    "symbolic load",
                )
            )
            return value
        value = self.fresh_value(dtype="int")
        self.emit(
            Handler.Const,
            [],
            value,
            attributes={"value": literal},
        )
        return value

    def _control_arithmetic_value(self, spelling: str) -> SSAValue | None:
        """Lower planner-authored scalar arithmetic to ordinary SSA.

        Dynamic loop bounds are retained as small expressions over resident
        ``u_control_<id>`` values (for example ``((u_control_6 -
        u_control_8) / u_control_4) + 1``).  They are compiler IR, not host
        Python to evaluate.  Parse only the closed arithmetic vocabulary the
        planner emits and return ``None`` for every other shape so the caller
        preserves its explicit symbolic-load shortfall.
        """

        try:
            expression = ast.parse(spelling, mode="eval").body
        except (SyntaxError, ValueError):
            return None

        binary_handlers = {
            ast.Add: Handler.Add,
            ast.Sub: Handler.Sub,
            ast.Mult: Handler.Mul,
            ast.Div: Handler.Div,
            ast.FloorDiv: Handler.FloorDiv,
            ast.Mod: Handler.Mod,
        }

        def lower(node: ast.AST) -> SSAValue | None:
            if isinstance(node, ast.Constant) and isinstance(
                node.value, (bool, int, float)
            ):
                value = self.fresh_value(
                    dtype="bool" if isinstance(node.value, bool)
                    else "int" if isinstance(node.value, int)
                    else "float64"
                )
                self.emit(
                    Handler.Const, [], value,
                    attributes={"value": node.value},
                )
                return value
            if isinstance(node, ast.Name):
                control = re.fullmatch(r"u_control_(\d+)", node.id)
                if control is not None:
                    return self.external_value(int(control.group(1)))
                value = re.fullmatch(r"value_(\d+)", node.id)
                if value is not None:
                    return self.external_value(int(value.group(1)))
                return self.uniform_values.get(node.id) or (
                    self.local_control_values.get(node.id)
                )
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                operand = lower(node.operand)
                if operand is None:
                    return None
                if isinstance(node.op, ast.UAdd):
                    return operand
                result = self.fresh_value(
                    dtype=operand.dtype,
                    shape=operand.shape,
                )
                self.emit(Handler.Neg, [operand], result)
                return result
            if isinstance(node, ast.BinOp):
                handler = binary_handlers.get(type(node.op))
                if handler is None:
                    return None
                left = lower(node.left)
                right = lower(node.right)
                if left is None or right is None:
                    return None
                result = self.fresh_value(
                    dtype=left.dtype or right.dtype,
                    shape=left.shape or right.shape,
                )
                self.emit(
                    handler,
                    [left, right],
                    result,
                    attributes={"binding": "control_expression"},
                )
                return result
            return None

        return lower(expression)

    def lower_control_expression(
        self,
        expression: ControlExpression,
        *,
        result_override: SSAValue | None = None,
    ) -> SSAValue:
        if expression.op == "value":
            return self.external_value(int(expression.value_id))
        if expression.op == "const":
            result = result_override or self.fresh_value(
                dtype="bool" if isinstance(expression.literal, bool) else None
            )
            self.emit(
                Handler.Const, [], result,
                attributes={"value": expression.literal},
            )
            return result
        if expression.op == "sequence_nonempty":
            if expression.value_id is None:
                raise ValueError("sequence truth predicate requires a value id")
            sequence_id = int(expression.value_id)
            descriptor = self._sequence_descriptor(
                sequence_id,
                policy=("unique" if bool(expression.literal) else "duplicates"),
                writable=True,
                location=f"control.sequence_nonempty[{sequence_id}]",
            )
            if descriptor is None:
                raise ValueError(
                    "cannot lower sequence truth predicate with conflicting "
                    f"storage policy for value {sequence_id}"
                )
            length_address = self.sequence_storage_values[sequence_id][
                len(descriptor.column_value_ids)
            ]
            length = self.fresh_value(dtype="int")
            self.emit(
                Handler.Load,
                [length_address],
                length,
                attributes={
                    "binding": "ssa_sequence_length",
                    "sequence_value_id": sequence_id,
                },
            )
            zero = self.constant_value(0)
            result = result_override or self.fresh_value(dtype="bool")
            self.emit(
                Handler.Gt,
                [length, zero],
                result,
                attributes={
                    "binding": "sequence_nonempty",
                    "sequence_value_id": sequence_id,
                },
            )
            return result
        operand_values = [
            self.lower_control_expression(operand)
            for operand in expression.operands
        ]
        if expression.op in {"float", "int", "bool"}:
            return operand_values[0]
        if expression.op == "item":
            source = operand_values[0]
            if not source.shape:
                return source
            index = self.constant_value(0)
            address = self.fresh_value(dtype="ptr")
            self.emit(
                Handler.GetElementPtr, [source, index], address,
                attributes={"binding": "control_scalar_item"},
            )
            result = result_override or self.fresh_value(dtype=source.dtype)
            self.emit(
                Handler.Load, [address], result,
                attributes={"binding": "control_scalar_item"},
            )
            return result
        handlers = {
            "add": Handler.Add, "sub": Handler.Sub,
            "mul": Handler.Mul, "div": Handler.Div,
            "neg": Handler.Neg,
            "lt": Handler.Lt, "le": Handler.Le,
            "gt": Handler.Gt, "ge": Handler.Ge,
            "eq": Handler.Eq, "ne": Handler.Ne,
            "and": Handler.LAnd, "or": Handler.LOr,
            "not": Handler.LNot,
            "bitand": Handler.And, "bitor": Handler.Or,
            "bitxor": Handler.Xor, "shl": Handler.Shl,
            "shr": Handler.Shr,
        }
        handler = handlers.get(expression.op)
        if handler is None:
            raise ValueError(
                f"unsupported control expression {expression.op!r}"
            )
        result = result_override or self.fresh_value(
            dtype=(
                "bool" if expression.op in {
                    "lt", "le", "gt", "ge", "eq", "ne",
                    "and", "or", "not",
                } else operand_values[0].dtype
            )
        )
        self.emit(handler, operand_values, result, attributes={
            "binding": "control_expression",
            "source_value_id": expression.value_id,
        })
        if expression.value_id is not None:
            self.external_values[int(expression.value_id)] = result
        return result

    def lower(self, block: ControlBlock, *, path: str = "root") -> None:
        previous = self._evolution_source
        if self.evolution is not None:
            self._evolution_source = self.evolution.component_for_artifact(block)
        try:
            self._lower(block, path=path)
        finally:
            self._evolution_source = previous

    def _pure_region_index(self, child: "ControlBlock") -> int | None:
        """The region a child emits, when reordering it is provably safe.

        Safe means: the child is a single scheduled-region statement, the
        region has a recorded signature to order by, and it carries no table
        or sequence operations -- those are EFFECTS, whose relative order the
        plan may be preserving for reasons dataflow over value ids cannot
        see. An effectful or unrecognised child is a reordering barrier.
        """

        if not isinstance(child, StatementBlock) or len(child.lines) != 1:
            return None
        match = _REGION_MARKER.fullmatch(str(child.lines[0]))
        if match is None:
            return None
        region_index = int(match.group(1))
        if region_index not in self.region_signatures:
            return None
        if self.table_region_operations.get(region_index):
            return None
        if self.table_region_post_operations.get(region_index):
            return None
        return region_index

    def _dependency_ordered(
        self, children: "tuple[ControlBlock, ...]"
    ) -> "list[ControlBlock]":
        """Reorder contiguous runs of pure region statements by dataflow.

        The planner's flat schedule can list a consumer region before its
        producer.  The lowering then reaches the consumer's feed before any
        instruction produces it, ``external_value`` mints the feed as a
        FORMAL, and the emitted function silently grows a parameter that is
        also a region output -- a one-parameter source becoming a
        two-parameter program with an unnamed formal no caller could fill
        (scorecard levels: nested calls, division-and-power).

        The signatures needed to order correctly are already here, so order
        by them: within each contiguous run of pure region statements, a
        stable topological sort placing producers before the regions that
        feed on them.  Stability preserves the plan's order wherever dataflow
        does not force a change, and anything effectful or unrecognised is a
        barrier that no region crosses.
        """

        ordered: list[ControlBlock] = []
        run: list[tuple[int, ControlBlock]] = []

        def flush() -> None:
            if len(run) < 2:
                ordered.extend(child for _index, child in run)
                run.clear()
                return
            produced_by = {
                int(output): position
                for position, (region_index, _child) in enumerate(run)
                for output in self.region_signatures[region_index][1]
            }
            remaining = list(range(len(run)))
            placed: set[int] = set()
            emitted_outputs: set[int] = set()
            while remaining:
                progressed = False
                for position in tuple(remaining):
                    region_index, _child = run[position]
                    feeds = self.region_signatures[region_index][0]
                    if all(
                        int(feed) not in produced_by
                        or produced_by[int(feed)] == position
                        or produced_by[int(feed)] in placed
                        for feed in feeds
                    ):
                        ordered.append(run[position][1])
                        placed.add(position)
                        remaining.remove(position)
                        progressed = True
                if not progressed:
                    # A dependency cycle among regions is not resolvable by
                    # ordering; emit the plan's own order for the remainder
                    # rather than looping forever or guessing.
                    ordered.extend(run[position][1] for position in remaining)
                    remaining.clear()
            run.clear()

        for child in children:
            region_index = self._pure_region_index(child)
            if region_index is None:
                flush()
                ordered.append(child)
            else:
                run.append((region_index, child))
        flush()
        return ordered

    def _lower(self, block: ControlBlock, *, path: str = "root") -> None:
        if isinstance(block, SequenceBlock):
            # Reordering is confined to straight-line context. Inside a loop
            # body the carried-value machinery is itself an ordering
            # constraint -- the LAST publish of a carried id becomes the
            # phi's latch operand via ``external_values`` rebinding -- and
            # that dependency is invisible to feed/output signatures, so the
            # sort must not run there.
            children = (
                tuple(block.blocks)
                if self.loop_targets
                else tuple(self._dependency_ordered(tuple(block.blocks)))
            )
            for index, child in enumerate(children):
                self.lower(child, path=f"{path}.sequence[{index}]")
            return
        if isinstance(block, StatementBlock):
            for index, line in enumerate(block.lines):
                match = _REGION_MARKER.fullmatch(str(line))
                location = f"{path}.statement[{index}]"
                callsite = _CALLSITE_MARKER.fullmatch(str(line))
                if callsite is not None:
                    self.emit_plan_callsite(
                        int(callsite.group(1)), location=location
                    )
                    continue
                if match is not None:
                    region_index = int(match.group(1))
                    memberships = self.declared_region_memberships.get(
                        region_index, ()
                    )
                    self.active_deployments.extend(memberships)
                    try:
                        self.emit_region_call(
                            region_index,
                            location=location,
                        )
                    finally:
                        if memberships:
                            del self.active_deployments[-len(memberships):]
                else:
                    self.emit(
                        Handler.Call,
                        [],
                        attributes={
                            "callee": "unlowered_control_statement",
                            "source": str(line),
                        },
                    )
                    self.shortfalls.append(
                        SSALoweringShortfall(
                            "control",
                            "statement",
                            location,
                            f"statement remains untranslated: {line!r}",
                        )
                    )
            return
        if isinstance(block, LoopBlock):
            self.lower_loop(block, path=path)
            return
        if isinstance(block, ConditionalBlock):
            self.lower_conditional(block, path=path)
            return
        if isinstance(block, WhileBlock):
            self.lower_while(block, path=path)
            return
        if isinstance(block, LoopControlBlock):
            if not self.loop_targets:
                raise ValueError(f"{block.action} appears outside a loop")
            latch, exit_block = self.loop_targets[-1]
            target = exit_block if block.action == "break" else latch
            if block.action == "break" and self.loop_exit_contexts:
                context = self.loop_exit_contexts[-1]
                carried_values = []
                for updated_id, initial_id, current in context["carried"]:
                    candidate = self.external_values.get(
                        int(updated_id),
                        self.external_values.get(int(initial_id), current),
                    )
                    carried_values.append(
                        candidate
                        if self._value_dominates_current_edge(candidate)
                        else current
                    )
                context["break_edges"].append((
                    self.current, tuple(carried_values)
                ))
            if block.predicate_value_id is None:
                self.branch(target)
                self.current.instrs[-1].attributes["source_control"] = (
                    block.source_action or block.action
                )
                self.current = self.new_block("unreachable_loop_control")
            else:
                fallthrough = self.new_block("loop_control_next")
                predicate = (
                    self.lower_control_expression(block.predicate_expression)
                    if block.predicate_expression is not None
                    else self.external_value(
                        block.predicate_value_id, dtype="bool"
                    )
                )
                self.conditional_branch(
                    predicate,
                    target if block.expect_true else fallthrough,
                    fallthrough if block.expect_true else target,
                )
                self.current.instrs[-1].attributes["source_control"] = (
                    block.source_action or block.action
                )
                self.current = fallthrough
            return
        if isinstance(block, CallBlock):
            # Hierarchy planning has already unified the bound value IDs.
            # CallBlock is lexical organization around the nested compiled
            # control, not an additional runtime invocation.
            self.lower(block.callee, path=f"{path}.callee")
            return
        if isinstance(block, ExternalReferenceCallBlock):
            arguments = [
                self.external_value(value_id, dtype="opaque_ref")
                for value_id in block.argument_value_ids
            ]
            arguments.extend(
                self.external_value(value_id, dtype="opaque_ref")
                for _name, value_id in block.keyword_argument_value_ids
            )
            result = SSAValue(
                int(block.result_value_id), dtype=str(block.result_dtype)
            )
            self.emit(
                Handler.Call,
                arguments,
                result,
                attributes={
                    "callee": "turing_external_reference_call",
                    "external_reference": True,
                    "external_identity": str(block.identity),
                    "external_callsite_id": int(block.callsite_id),
                    "external_domain": str(block.external_domain),
                    "shell_abi": str(block.shell_abi),
                    "native_abi": str(block.native_abi),
                    "runtime_owner": str(block.runtime_owner),
                    "shell_profiles": tuple(block.shell_profiles),
                    "keyword_names": tuple(
                        name for name, _value_id
                        in block.keyword_argument_value_ids
                    ),
                    "argument_frame": "turing.external-reference-arguments.v1",
                    "result_frame": "turing.external-reference-value.v1",
                    "object_policy": "shell-owned-opaque-handles",
                    "extraction_identity": str(block.identity),
                },
            )
            self.external_values[int(block.result_value_id)] = result
            return
        if isinstance(block, ValidationBlock):
            self.validation_contracts.append({
                "predicate_value_id": int(block.predicate_value_id),
                "error_code": int(block.error_code),
                "expect_true": bool(block.expect_true),
                "predicate_expression": _control_expression_mapping(
                    block.predicate_expression
                ),
                "extraction_identity": block.extraction_identity,
            })
            predicate = (
                self.lower_control_expression(block.predicate_expression)
                if block.predicate_expression is not None
                else self.external_value(
                    block.predicate_value_id,
                    dtype="bool",
                )
            )
            passed = self.new_block("validation_pass")
            failed = self.new_block("validation_fail")
            if block.expect_true:
                self.conditional_branch(predicate, passed, failed)
            else:
                self.conditional_branch(predicate, failed, passed)
            self.current = failed
            self.emit(
                Handler.Call,
                [],
                attributes={
                    "callee": "turing_validation_error",
                    "error_code": int(block.error_code),
                    "extraction_identity": block.extraction_identity,
                },
            )
            self.branch(passed)
            self.current = passed
            return
        if isinstance(block, SequenceMutationBlock):
            self.lower_sequence_mutation(block.mutation, path=path)
            return
        if isinstance(block, SequenceQueryBlock):
            self.lower_sequence_query(block, path=path)
            return
        if isinstance(block, StreamPublishBlock):
            args = [self.external_value(block.value_id)]
            if block.count_value_id is not None:
                args.append(self.external_value(block.count_value_id))
            if block.predicate_value_id is not None:
                args.append(
                    self.external_value(
                        block.predicate_value_id,
                        dtype="bool",
                    )
                )
            self.emit(
                Handler.Call,
                args,
                attributes={
                    "callee": "turing_stream_publish",
                    "stream_id": int(block.stream_id),
                    "final": bool(block.final),
                },
            )
            return
        if isinstance(block, StateMachineTick):
            self.lower_state_machine(block, path=path)
            return
        if isinstance(block, ParallelDeployment):
            deployment_id = self.next_deployment_id
            self.next_deployment_id += 1
            record = {
                "region_id": deployment_id,
                "kind": "parallel_candidate",
                "schedule": "independent_lanes",
                "schedule_preference": block.schedule_preference,
                "lane_count": len(block.lanes),
                "iteration_space": None,
                "carried_aliases": (),
                "recursion_region_id": None,
                "origin": "parallel_block",
                "source_loop_node_id": None,
                "scale": 1,
                "join": DeploymentJoin(),
                "declared_lanes": (),
            }
            self.deployment_records.append(record)
            self.emit_deployment_boundary(Handler.Deploy, record)
            for index, lane in enumerate(block.lanes):
                self.active_deployments.append((deployment_id, index))
                try:
                    self.lower(lane, path=f"{path}.lane[{index}]")
                finally:
                    self.active_deployments.pop()
            self.emit_deployment_boundary(Handler.Join, record)
            return
        raise TypeError(f"unknown control block {type(block).__name__}")

    def _sequence_descriptor(
        self,
        value_id: int,
        *,
        policy: str,
        writable: bool,
        location: str,
        column_count: int | None = None,
        retains_deleted_rows: bool = False,
        nested_table: bool = False,
        nested_value_dtype: str | None = None,
        element_dtype: str | None = None,
        column_dtypes: tuple[str | None, ...] = (),
    ) -> SSASequenceDescriptor | None:
        value_id = int(value_id)
        resolved = self.resolved_sequence_schemas.get(value_id)
        existing = self.sequence_descriptors.get(value_id)
        if column_count is None:
            column_count = (
                int(resolved.column_count) if resolved is not None
                else len(existing.column_value_ids) if existing is not None
                else 1
            )
        if resolved is not None:
            if (
                int(resolved.column_count) != int(column_count)
                or str(resolved.policy) != str(policy)
            ):
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence", "resolved-schema-mismatch", location,
                    f"sequence value {value_id} disagrees with its "
                    "whole-program resolved schema: this function sees "
                    f"{int(column_count)} column(s)/policy {policy!r}, "
                    f"but the survey resolved {int(resolved.column_count)} "
                    f"column(s)/policy {resolved.policy!r} from every "
                    "function that touches this sequence",
                ))
                return None
            # writable/retains_deleted_rows/nested_table are resolved as an
            # OR across every function during the survey -- this function's
            # own local view is always a subset of that, never a conflict,
            # so the resolved value simply wins rather than being checked.
            writable = bool(writable or resolved.writable)
            retains_deleted_rows = bool(
                retains_deleted_rows or resolved.retains_deleted_rows
            )
            nested_table = bool(nested_table or resolved.nested_table)
            nested_value_dtype = nested_value_dtype or resolved.nested_value_dtype
        key_columns = (
            tuple(range(max(1, int(column_count) - 1)))
            if policy == "unique" else ()
        )
        if existing is not None:
            if existing.key_columns != key_columns:
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence",
                    "conflicting-policy",
                    location,
                    f"sequence value {value_id} is used with both unique and duplicate policies",
                ))
                return None
            if len(existing.column_value_ids) != int(column_count):
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence",
                    "conflicting-width",
                    location,
                    f"sequence value {value_id} is used with both "
                    f"{len(existing.column_value_ids)} and {int(column_count)} columns",
                ))
                return None
            if retains_deleted_rows and existing.live_flags_value_id is None:
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence", "conflicting-retention", location,
                    f"sequence value {value_id} requires deletion retention "
                    "after storage was declared without live flags",
                ))
                return None
            return existing
        def _canonicalize_column(raw_dtype: str | None, column: int) -> str | None:
            canonical = _canonical_sequence_dtype(raw_dtype)
            if canonical != raw_dtype and _LOG.isEnabledFor(logging.DEBUG):
                _LOG.debug(
                    "sequence %d column %d at %s: widened dtype %r -> %r "
                    "(this array is shared across every function that "
                    "touches the sequence, so its element width must be "
                    "one fact, not whatever the first function to see it "
                    "happened to infer)",
                    value_id, column, location, raw_dtype, canonical,
                )
            return canonical

        first_dtype = _canonicalize_column(element_dtype or (
            column_dtypes[0] if column_dtypes else None
        ), 0)
        data = self.external_value(value_id, dtype=first_dtype)
        if first_dtype is not None and element_dtype is not None:
            # An explicit element contract (literal bytes, joined outer-row
            # handles, fill materialization) is stronger than the provisional
            # float dtype assigned when a structural constant first entered
            # the control value map.
            data.dtype = str(first_dtype)
        extra_columns = tuple(
            self.fresh_value(dtype=_canonicalize_column(
                (
                    "int64"
                    if nested_table and index == int(column_count) - 2
                    else str(
                        column_dtypes[index + 1]
                        if index + 1 < len(column_dtypes)
                        and column_dtypes[index + 1] is not None
                        else data.dtype or "unknown"
                    )
                ),
                index + 1,
            ))
            for index in range(max(0, int(column_count) - 1))
        )
        self.arguments.extend(extra_columns)
        # Mutable scalar cells are one-element typed arenas at the C ABI, not
        # opaque pointer-typed scalars (which the Fortran emitter would have
        # no element type for and could accidentally pass by value).
        # int64, not the generic "int" default: these two cells are shared
        # ABI storage a sequence-helper caller may bind to an externally
        # int64-typed value (e.g. a keyed instance field's own length, see
        # _storage_values in ir_sequence_tables.py) -- both sides of that
        # shared value must declare the same width or the Fortran call
        # fails with a real ABI type mismatch, not a cosmetic one.
        length_address = self.fresh_value(dtype="int64", shape=(1,))
        capacity = self.fresh_value(dtype="int64")
        requires_status = bool(writable) or policy == "unique"
        status_address = (
            self.fresh_value(dtype="int", shape=(1,))
            if requires_status else None
        )
        self.arguments.extend((length_address, capacity))
        if status_address is not None:
            self.arguments.append(status_address)
        live_flags = (
            self.fresh_value(dtype="bool") if retains_deleted_rows else None
        )
        if live_flags is not None:
            self.arguments.append(live_flags)
        child_table_pool = None
        child_pool_values: tuple[SSAValue, ...] = ()
        if nested_table:
            child_keys = self.fresh_value(dtype="unknown")
            child_values = self.fresh_value(
                dtype=nested_value_dtype or "unknown"
            )
            child_lengths = self.fresh_value(dtype="int")
            child_capacity = self.fresh_value(dtype="int")
            child_stride = self.fresh_value(dtype="int")
            child_status = self.fresh_value(dtype="int")
            child_live = self.fresh_value(dtype="bool")
            child_pool_values = (
                child_keys, child_values, child_lengths, child_capacity,
                child_stride, child_status, child_live,
            )
            self.arguments.extend(child_pool_values)
            self.external_values.update(
                (int(value.id), value) for value in child_pool_values
            )
            child_table_pool = SSAChildTablePoolDescriptor(
                handle_column=1,
                column_value_ids=(int(child_keys.id), int(child_values.id)),
                length_value_id=int(child_lengths.id),
                capacity_value_id=int(child_capacity.id),
                row_stride_value_id=int(child_stride.id),
                status_value_id=int(child_status.id),
                live_flags_value_id=int(child_live.id),
                column_dtypes=(
                    "unknown", nested_value_dtype or "unknown"
                ),
                key_columns=(0,),
                writable=bool(writable),
            )
        descriptor = SSASequenceDescriptor(
            sequence_id=value_id,
            column_value_ids=(int(data.id), *(
                int(column.id) for column in extra_columns
            )),
            length_address_id=int(length_address.id),
            capacity_value_id=int(capacity.id),
            status_address_id=(
                None if status_address is None else int(status_address.id)
            ),
            column_dtypes=(
                str(data.dtype or "unknown"),
                *(str(column.dtype or "unknown") for column in extra_columns),
            ),
            key_columns=key_columns,
            live_flags_value_id=(
                None if live_flags is None else int(live_flags.id)
            ),
            capacity_policy=SSASequenceCapacityPolicy.FIXED,
            writable=bool(writable),
            child_table_pool=child_table_pool,
        )
        self.sequence_descriptors[value_id] = descriptor
        self.sequence_storage_values[value_id] = (
            data, *extra_columns, length_address, capacity,
            *((live_flags,) if live_flags is not None else ()),
        )
        if status_address is not None:
            self.sequence_status_values[value_id] = status_address
        return descriptor

    def _sequence_status_address(self, status_arena: SSAValue) -> SSAValue:
        zero = self.constant_value(0)
        address = self.fresh_value(dtype="ptr")
        self.emit(
            Handler.GetElementPtr,
            [status_arena, zero],
            address,
            attributes={"binding": "ssa_sequence_status"},
        )
        return address

    def lower_sequence_mutation(
        self,
        mutation: ControlSequenceMutation,
        *,
        path: str,
    ) -> None:
        if mutation.predicate_expression is None:
            self._lower_sequence_mutation_body(mutation, path=path)
            return
        predicate = self.lower_control_expression(
            mutation.predicate_expression
        )
        selected = self.new_block("sequence_mutation_selected")
        skipped = self.new_block("sequence_mutation_skipped")
        complete = self.new_block("sequence_mutation_merge")
        self.conditional_branch(predicate, selected, skipped)
        self.current = skipped
        self.branch(complete)
        self.current = selected
        self._lower_sequence_mutation_body(mutation, path=path)
        if not self.current.successors:
            self.branch(complete)
        self.current = complete

    def lower_sequence_query(
        self,
        query: SequenceQueryBlock,
        *,
        path: str,
    ) -> None:
        """Lower a lexical query against the sequence's length-cell ABI."""

        location = f"{path}.sequence_query[{query.source_call_node_id}]"
        if query.operation == "lookup":
            self._emit_table_lookup(
                int(query.result_value_id),
                (
                    int(query.key_value_ids[0])
                    if len(query.key_value_ids) == 1
                    else tuple(map(int, query.key_value_ids))
                ),
                int(query.sequence_value_id),
            )
            return
        descriptor = self._sequence_descriptor(
            int(query.sequence_value_id),
            policy="duplicates",
            writable=True,
            location=location,
        )
        if descriptor is None:
            return
        storage = self.sequence_storage_values[int(query.sequence_value_id)]
        length_address = storage[len(descriptor.column_value_ids)]
        attributes = {
            "binding": f"ssa_sequence_{query.operation}",
            "sequence_value_id": int(query.sequence_value_id),
            "source_call_node_id": query.source_call_node_id,
            "extraction_identity": query.extraction_identity,
        }
        length = self.fresh_value(dtype="int64")
        self.emit(Handler.Load, [length_address], length, attributes=attributes)
        if query.operation == "length":
            result = self.produced_value(
                int(query.result_value_id),
                dtype="int64",
                claim_provisional_definition=True,
            )
            self.emit(
                Handler.Cast,
                [length],
                result,
                attributes={**attributes, "target_dtype": "int64"},
            )
            self.external_values[int(query.result_value_id)] = result
            for alias_id in query.result_alias_ids:
                self.external_values[int(alias_id)] = result
            return

        zero = self.constant_value(0)
        nonempty = self.fresh_value(dtype="bool")
        self.emit(Handler.Gt, [length, zero], nonempty, attributes=attributes)
        selected = self.new_block("sequence_query_selected")
        defaulted = self.new_block("sequence_query_defaulted")
        complete = self.new_block("sequence_query_merge")
        self.conditional_branch(nonempty, selected, defaulted)

        self.current = selected
        address = self.fresh_value(dtype="ptr")
        self.emit(
            Handler.GetElementPtr,
            [storage[0], zero],
            address,
            attributes=attributes,
        )
        selected_value = self.fresh_value(
            dtype=("int" if query.row_handle else storage[0].dtype)
        )
        self.emit(Handler.Load, [address], selected_value, attributes=attributes)
        self.branch(complete)

        self.current = defaulted
        default_value = (
            self.constant_value(-1)
            if query.row_handle
            else self.external_value(int(query.default_value_id))
        )
        self.branch(complete)

        self.current = complete
        result = self.produced_value(
            int(query.result_value_id),
            dtype=str(selected_value.dtype or default_value.dtype or "unknown"),
            claim_provisional_definition=True,
        )
        self.emit(
            Handler.Phi,
            [selected_value, default_value],
            result,
            attributes={
                **attributes,
                "incoming_blocks": (selected.name, defaulted.name),
            },
        )
        self.external_values[int(query.result_value_id)] = result
        for alias_id in query.result_alias_ids:
            self.external_values[int(alias_id)] = result

    def ensure_plan_callsite_result(
        self, value_id: int, *, location: str,
    ) -> None:
        """Materialize a planned call at its first lexical structural use."""

        callsite_id = self.plan_callsite_by_result.get(int(value_id))
        if (
            callsite_id is None
            or int(callsite_id) in self.emitted_plan_callsites
        ):
            return
        self.emit_plan_callsite(int(callsite_id), location=location)

    def _lower_sequence_mutation_body(
        self,
        mutation: ControlSequenceMutation,
        *,
        path: str,
    ) -> None:
        operation = str(mutation.operator)
        location = f"{path}.sequence_mutation[{mutation.effect_node_id}]"
        if mutation.policy is None:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence",
                operation,
                location,
                "destination uniqueness policy is unresolved",
            ))
            return
        if not mutation.argument_value_ids and operation not in {"clear", "pop"}:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence",
                operation,
                location,
                "sequence mutation has no explicit source value",
            ))
            return
        destination = self._sequence_descriptor(
            mutation.sequence_value_id,
            policy=mutation.policy,
            writable=True,
            location=location,
        )
        if destination is None:
            return

        if operation == "clear" and not mutation.argument_kind.startswith(
            "mapping_"
        ):
            if mutation.argument_value_ids:
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence", operation, location,
                    "resident sequence clear takes no arguments",
                ))
                return
            storage = self.sequence_storage_values[
                int(destination.sequence_id)
            ]
            length_address = storage[len(destination.column_value_ids)]
            self.emit(
                Handler.Store,
                [self.constant_value(0), length_address],
                attributes={
                    "binding": "ssa_sequence_clear",
                    "sequence_id": int(destination.sequence_id),
                    "source_effect_node_id": int(mutation.effect_node_id),
                    **({
                        "extraction_identity": str(
                            mutation.extraction_identity
                        ),
                    } if mutation.extraction_identity is not None else {}),
                },
            )
            return

        if operation == "pop" and not mutation.argument_kind.startswith(
            "mapping_"
        ):
            if mutation.argument_value_ids:
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence", operation, location,
                    "resident sequence pop currently supports only the "
                    "zero-argument last-row operation",
                ))
                return
            if len(destination.column_value_ids) != 1:
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence", operation, location,
                    "resident sequence pop requires one resolved value "
                    "column before aggregate-row results are admitted",
                ))
                return
            storage = self.sequence_storage_values[
                int(destination.sequence_id)
            ]
            length_address = storage[len(destination.column_value_ids)]
            attributes = {
                "binding": "ssa_sequence_pop",
                "sequence_id": int(destination.sequence_id),
                "source_effect_node_id": int(mutation.effect_node_id),
                **({
                    "extraction_identity": str(
                        mutation.extraction_identity
                    ),
                } if mutation.extraction_identity is not None else {}),
            }
            length = self.fresh_value(dtype="int64")
            self.emit(
                Handler.Load, [length_address], length,
                attributes=attributes,
            )
            nonempty = self.fresh_value(dtype="bool")
            self.emit(
                Handler.Gt,
                [length, self.constant_value(0)],
                nonempty,
                attributes=attributes,
            )
            selected = self.new_block("sequence_pop_selected")
            empty = self.new_block("sequence_pop_empty")
            self.conditional_branch(nonempty, selected, empty)

            self.current = empty
            self.emit(
                Handler.Call,
                [],
                attributes={
                    "callee": "turing_validation_error",
                    "error_code": 71,
                    **attributes,
                },
            )
            self.branch(selected)

            self.current = selected
            new_length = self.fresh_value(dtype="int64")
            self.emit(
                Handler.Sub,
                [length, self.constant_value(1)],
                new_length,
                attributes=attributes,
            )
            address = self.fresh_value(dtype="ptr")
            self.emit(
                Handler.GetElementPtr,
                [storage[0], new_length],
                address,
                attributes=attributes,
            )
            result = self.produced_value(
                int(mutation.effect_node_id),
                dtype=str(storage[0].dtype or "unknown"),
                claim_provisional_definition=True,
            )
            self.emit(
                Handler.Load, [address], result,
                attributes=attributes,
            )
            self.emit(
                Handler.Store,
                [new_length, length_address],
                attributes=attributes,
            )
            self.external_values[int(mutation.effect_node_id)] = result
            return

        if mutation.argument_kind.startswith("mapping_"):
            if operation == "update":
                if (
                    mutation.argument_kind != "mapping_items"
                    or len(mutation.argument_value_ids) % 2
                ):
                    self.shortfalls.append(SSALoweringShortfall(
                        "ssa-table", operation, location,
                        "mapping update requires deterministic key/value "
                        "pairs from a fixed mapping aggregate",
                    ))
                    return
                value_columns = tuple(
                    column
                    for column in range(len(destination.column_value_ids))
                    if column not in destination.key_columns
                )
                if not destination.key_columns or len(value_columns) != 1:
                    self.shortfalls.append(SSALoweringShortfall(
                        "ssa-table", operation, location,
                        "mapping update requires a resolved table schema "
                        "with key columns and one value column",
                    ))
                    return
                for key_id, value_id in zip(
                    mutation.argument_value_ids[0::2],
                    mutation.argument_value_ids[1::2],
                ):
                    self._emit_table_store(
                        int(mutation.effect_node_id), int(key_id),
                        int(value_id), int(destination.sequence_id),
                    )
                return
            if operation == "pop":
                optional_none = (
                    mutation.argument_kind == "mapping_pop_default_none"
                )
                if len(mutation.argument_value_ids) != (2 if optional_none else 1):
                    self.shortfalls.append(SSALoweringShortfall(
                        "ssa-table", operation, location,
                        "mapping pop with a default requires an explicit "
                        "optional-result ABI before lookup-and-delete; "
                        f"child_pool={destination.child_table_pool is not None}, "
                        f"key_columns={destination.key_columns}, "
                        f"column_dtypes={destination.column_dtypes}, "
                        f"arguments={mutation.argument_value_ids}, "
                        f"resolved_schema={self.resolved_sequence_schemas.get(int(destination.sequence_id))!r}",
                    ))
                    return
                key_id = int(mutation.argument_value_ids[0])
                if optional_none:
                    self.table_lookup_defaults[int(mutation.effect_node_id)] = -1
                self._emit_table_lookup(
                    int(mutation.effect_node_id), key_id,
                    int(destination.sequence_id),
                )
                self._emit_table_delete(
                    int(mutation.effect_node_id), key_id,
                    int(destination.sequence_id),
                    f"sequence:{int(destination.sequence_id)}",
                )
                return
            if operation == "setdefault":
                pool = destination.child_table_pool
                key_count = len(destination.key_columns)
                if (
                    pool is None
                    or len(mutation.argument_value_ids) != key_count + 1
                ):
                    self.shortfalls.append(SSALoweringShortfall(
                        "ssa-table", operation, location,
                        "mapping setdefault requires a resolved child-table "
                        "pool, exact key columns, and one aggregate default; "
                        f"child_pool={pool is not None}, "
                        f"key_columns={destination.key_columns}, "
                        f"arguments={mutation.argument_value_ids}, "
                        f"resolved_schema={self.resolved_sequence_schemas.get(int(destination.sequence_id))!r}",
                    ))
                    return
                from .ir_sequence_tables import lower_table_lookup

                lookup_name = (
                    f"ssa_sequence_{destination.sequence_id}_setdefault_lookup"
                )
                lookup = lower_table_lookup(
                    destination,
                    function_name=lookup_name,
                    first_value_id=self.next_value_id,
                )
                self._register_sequence_lowering(lookup)
                lookup_handle = self.fresh_value(dtype="int")
                key_ids = tuple(map(
                    int, mutation.argument_value_ids[:key_count]
                ))
                self.emit(
                    Handler.Call,
                    [
                        *self.sequence_storage_values[
                            int(destination.sequence_id)
                        ],
                        self.sequence_status_values[
                            int(destination.sequence_id)
                        ],
                        *self._table_query_values(key_ids),
                    ],
                    lookup_handle,
                    attributes={
                        "callee": lookup_name,
                        "source_linked": True,
                        "ssa_sequence_operation": "setdefault_lookup",
                        "sequence_id": int(destination.sequence_id),
                    },
                )
                status = self.fresh_value(dtype="int")
                self.emit(
                    Handler.Load,
                    [self._sequence_status_address(
                        self.sequence_status_values[
                            int(destination.sequence_id)
                        ]
                    )],
                    status,
                    attributes={"binding": "ssa_mapping_setdefault_status"},
                )
                found = self.fresh_value(dtype="bool")
                self.emit(
                    Handler.Gt, [status, self.constant_value(0)], found,
                    attributes={"binding": "ssa_mapping_setdefault_found"},
                )
                found_block = self.new_block("mapping_setdefault_found")
                missing_block = self.new_block("mapping_setdefault_missing")
                complete = self.new_block("mapping_setdefault_merge")
                self.conditional_branch(found, found_block, missing_block)

                self.current = found_block
                self.branch(complete)

                self.current = missing_block
                outer_length_address = self.sequence_storage_values[
                    int(destination.sequence_id)
                ][len(destination.column_value_ids)]
                outer_length = self.fresh_value(dtype="int64")
                self.emit(
                    Handler.Load, [outer_length_address], outer_length,
                    attributes={"binding": "ssa_mapping_setdefault_handle"},
                )
                new_handle = self.fresh_value(dtype="int")
                self.emit(
                    Handler.Cast, [outer_length], new_handle,
                    attributes={"target_dtype": "int"},
                )
                self.external_values[int(new_handle.id)] = new_handle
                child_length_address = self.fresh_value(dtype="ptr")
                self.emit(
                    Handler.GetElementPtr,
                    [
                        self.external_value(int(pool.length_value_id)),
                        new_handle,
                    ],
                    child_length_address,
                    attributes={"binding": "ssa_child_table_length"},
                )
                self.emit(
                    Handler.Store,
                    [self.constant_value(0), child_length_address],
                    attributes={"binding": "ssa_child_table_initialize"},
                )
                if pool.status_value_id is not None:
                    child_status_address = self.fresh_value(dtype="ptr")
                    self.emit(
                        Handler.GetElementPtr,
                        [
                            self.external_value(int(pool.status_value_id)),
                            new_handle,
                        ],
                        child_status_address,
                        attributes={"binding": "ssa_child_table_status"},
                    )
                    self.emit(
                        Handler.Store,
                        [self.constant_value(0), child_status_address],
                        attributes={"binding": "ssa_child_table_initialize"},
                    )
                self._emit_table_store(
                    int(mutation.effect_node_id), key_ids,
                    int(new_handle.id), int(destination.sequence_id),
                )
                self.branch(complete)

                self.current = complete
                result = self.produced_value(
                    int(mutation.effect_node_id),
                    dtype="int",
                    claim_provisional_definition=True,
                )
                self.emit(
                    Handler.Phi,
                    [lookup_handle, new_handle],
                    result,
                    attributes={
                        "incoming_blocks": (
                            found_block.name, missing_block.name
                        ),
                        "binding": "ssa_mapping_setdefault_result",
                    },
                )
                self.external_values[int(mutation.effect_node_id)] = result
                self.child_table_selections[int(mutation.effect_node_id)] = (
                    pool, result
                )
                return
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-table", operation, location,
                "unknown mapping mutation operator",
            ))
            return

        from .ir_sequence_tables import (
            lower_sequence_add,
            lower_sequence_append,
            lower_sequence_extend,
        )

        call_arguments: tuple[SSAValue, ...]
        deferred_record_row: tuple[int, str, int] | None = None
        if operation in {"append", "add"}:
            expected_columns = len(destination.column_value_ids)
            if len(mutation.argument_value_ids) != expected_columns:
                record_identity = self.sequence_record_identities.get(
                    int(destination.sequence_id)
                )
                if (
                    record_identity is not None
                    and len(mutation.argument_value_ids) == 1
                    and expected_columns > 1
                ):
                    deferred_record_row = (
                        int(mutation.argument_value_ids[0]),
                        str(record_identity),
                        int(expected_columns),
                    )
                else:
                    self.shortfalls.append(SSALoweringShortfall(
                        "ssa-sequence", operation, location,
                        "row insertion requires one explicit value per resident "
                        f"column; expected {expected_columns}, received "
                        f"{len(mutation.argument_value_ids)}",
                    ))
                    return
            joined_flat_id = self.joined_flat_sequence_ids.get(
                int(destination.sequence_id)
            )
            if joined_flat_id is not None:
                source_id = int(mutation.argument_value_ids[0])
                self.ensure_plan_callsite_result(
                    source_id, location=f"{location}.joined_source"
                )
                flat = self.sequence_descriptors[int(joined_flat_id)]
                singleton_value_id = self.joined_singleton_values.get(
                    source_id
                )
                source = (
                    None
                    if singleton_value_id is not None
                    else self._sequence_descriptor(
                        source_id,
                        policy="duplicates",
                        writable=False,
                        location=f"{location}.joined_source",
                    )
                )
                if singleton_value_id is None and source is None:
                    return
                count_name = f"ssa_sequence_{destination.sequence_id}_append"
                count_lowering = lower_sequence_append(
                    destination,
                    function_name=count_name,
                    first_value_id=self.next_value_id,
                )
                if not count_lowering.complete:
                    self.shortfalls.extend(
                        SSALoweringShortfall(
                            "ssa-sequence", item.code.value, location,
                            item.reason,
                        )
                        for item in count_lowering.shortfalls
                    )
                    return
                self._register_sequence_lowering(count_lowering)
                flat_name = (
                    f"ssa_sequence_{joined_flat_id}_append_singleton_"
                    f"{source_id}"
                    if singleton_value_id is not None
                    else f"ssa_sequence_{joined_flat_id}_extend_"
                    f"{source.sequence_id}"
                )
                flat_lowering = (
                    lower_sequence_append(
                        flat,
                        function_name=flat_name,
                        first_value_id=self.next_value_id,
                    )
                    if singleton_value_id is not None
                    else lower_sequence_extend(
                        flat,
                        source,
                        function_name=flat_name,
                        first_value_id=self.next_value_id,
                    )
                )
                if not flat_lowering.complete:
                    self.shortfalls.extend(
                        SSALoweringShortfall(
                            "ssa-sequence", item.code.value, location,
                            item.reason,
                        )
                        for item in flat_lowering.shortfalls
                    )
                    return
                self._register_sequence_lowering(flat_lowering)
                count_status = self.fresh_value(dtype="int")
                self.emit(
                    Handler.Call,
                    [
                        *self.sequence_storage_values[
                            destination.sequence_id
                        ],
                        self.constant_value(0),
                    ],
                    count_status,
                    attributes={
                        "callee": count_lowering.functions[-1].name,
                        "source_linked": True,
                        "ssa_sequence_operation": "append_joined_count",
                        "sequence_id": int(destination.sequence_id),
                        "source_effect_node_id": int(mutation.effect_node_id),
                    },
                )
                flat_status = self.fresh_value(dtype="int")
                joined_call_values = {
                    int(value.id): value
                    for value in (
                        *self.sequence_storage_values[int(joined_flat_id)],
                        *(
                            (self.external_value(singleton_value_id),)
                            if singleton_value_id is not None
                            else self.sequence_storage_values[
                                source.sequence_id
                            ]
                        ),
                    )
                }
                self.emit(
                    Handler.Call,
                    list(joined_call_values.values()),
                    flat_status,
                    attributes={
                        "callee": flat_lowering.functions[-1].name,
                        "source_linked": True,
                        "ssa_sequence_operation": (
                            "append_joined_singleton"
                            if singleton_value_id is not None
                            else "extend_joined_bytes"
                        ),
                        "sequence_id": int(joined_flat_id),
                        "joined_outer_sequence_id": int(
                            destination.sequence_id
                        ),
                        "joined_source_sequence_id": int(source_id),
                        "joined_source_value_id": (
                            None if singleton_value_id is None
                            else int(singleton_value_id)
                        ),
                        "source_effect_node_id": int(mutation.effect_node_id),
                    },
                )
                for sequence_id, status in (
                    (int(destination.sequence_id), count_status),
                    (int(joined_flat_id), flat_status),
                ):
                    self.emit(
                        Handler.Store,
                        [
                            status,
                            self._sequence_status_address(
                                self.sequence_status_values[sequence_id]
                            ),
                        ],
                        attributes={"binding": "ssa_sequence_status"},
                    )
                return
            function_name = (
                f"ssa_sequence_{destination.sequence_id}_{operation}"
            )
            lowering = (
                lower_sequence_append(
                    destination,
                    function_name=function_name,
                    first_value_id=self.next_value_id,
                )
                if operation == "append"
                else lower_sequence_add(
                    destination,
                    function_name=function_name,
                    first_value_id=self.next_value_id,
                )
            )
            mutation_values = tuple(
                (
                    self.lower_control_expression(expression)
                    if expression is not None
                    else self.external_value(value_id)
                )
                for value_id, expression in zip(
                    mutation.argument_value_ids,
                    (
                        *mutation.argument_expressions,
                        *(None for _ in range(
                            len(mutation.argument_value_ids)
                            - len(mutation.argument_expressions)
                        )),
                    ),
                )
            )
            if deferred_record_row is None:
                for mutation_value, element_dtype in zip(
                    mutation_values, destination.column_dtypes
                ):
                    if str(element_dtype) not in {"", "unknown", "None"}:
                        mutation_value.dtype = str(element_dtype)
            call_arguments = (
                *self.sequence_storage_values[destination.sequence_id],
                *mutation_values,
            )
        elif operation in {"extend", "replace"}:
            if mutation.argument_kind == "generator":
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence",
                    operation,
                    location,
                    "generator-backed extend requires the SSA iterator contract",
                ))
                return
            if mutation.argument_kind == "filtered_sequence":
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence",
                    operation,
                    location,
                    "filtered list-comprehension extend requires predicated compact materialization before eager insertion",
                ))
                return
            if len(mutation.argument_value_ids) != 1:
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence", operation, location,
                    "extend currently requires one resident source sequence",
                ))
                return
            source_id = int(mutation.argument_value_ids[0])
            self.ensure_plan_callsite_result(
                source_id, location=f"{location}.source"
            )
            source = self._sequence_descriptor(
                source_id,
                policy="duplicates",
                writable=False,
                location=location,
            )
            if source is None:
                return
            if operation == "replace":
                # A branch-selected sequence assignment keeps one stable
                # destination arena.  Clear its logical contents before the
                # ordinary resident-to-resident copy; capacity and status
                # remain owned by the destination descriptor.
                destination_storage = self.sequence_storage_values[
                    destination.sequence_id
                ]
                destination_length = destination_storage[
                    len(destination.column_value_ids)
                ]
                self.emit(
                    Handler.Store,
                    [self.constant_value(0), destination_length],
                    attributes={
                        "binding": "ssa_sequence_replace_clear",
                        "sequence_id": int(destination.sequence_id),
                    },
                )
            function_name = (
                f"ssa_sequence_{destination.sequence_id}_{operation}_"
                f"{source.sequence_id}"
            )
            lowering = lower_sequence_extend(
                destination,
                source,
                function_name=function_name,
                first_value_id=self.next_value_id,
            )
            call_arguments = tuple({
                value.id: value
                for value in (
                    *self.sequence_storage_values[destination.sequence_id],
                    *self.sequence_storage_values[source.sequence_id],
                )
            }.values())
        else:
            self.shortfalls.append(SSALoweringShortfall(
                "ssa-sequence", operation, location,
                "unknown sequence mutation operator",
            ))
            return

        if not lowering.complete:
            self.shortfalls.extend(
                SSALoweringShortfall(
                    "ssa-sequence",
                    item.code.value,
                    location,
                    item.reason,
                )
                for item in lowering.shortfalls
            )
            return
        self._register_sequence_lowering(lowering)
        callee = lowering.functions[-1].name
        status = self.fresh_value(dtype="int")
        self.emit(
            Handler.Call,
            list(call_arguments),
            status,
            attributes={
                "callee": callee,
                "source_linked": True,
                "ssa_sequence_operation": operation,
                "sequence_id": int(destination.sequence_id),
                "source_effect_node_id": int(mutation.effect_node_id),
                **({
                    "ssa_deferred_record_row": deferred_record_row,
                } if deferred_record_row is not None else {}),
                **({
                    "extraction_identity": str(mutation.extraction_identity),
                } if mutation.extraction_identity is not None else {}),
            },
        )
        self.emit(
            Handler.Store,
            [
                status,
                self._sequence_status_address(
                    self.sequence_status_values[destination.sequence_id]
                ),
            ],
            attributes={"binding": "ssa_sequence_status"},
        )

    def lower_conditional(
        self, conditional: ConditionalBlock, *, path: str,
    ) -> None:
        """Lower one authored branch and publish its lexical merge values."""

        predicate_sequence_id = None
        if conditional.entry_record_projections:
            handle_id = int(conditional.entry_record_projections[0][1])
            predicate = self.fresh_value(dtype="bool")
            self.emit(
                Handler.Ge,
                [self.external_value(handle_id, dtype="int"), self.constant_value(0)],
                predicate,
                attributes={
                    "binding": "optional_record_present",
                    "row_handle_value_id": handle_id,
                },
            )
        elif (
            conditional.predicate_expression is not None
            and conditional.predicate_expression.op == "value"
            and conditional.predicate_expression.value_id is not None
            and int(conditional.predicate_expression.value_id)
            in self.sequence_descriptors
        ):
            predicate_sequence_id = int(
                conditional.predicate_expression.value_id
            )
        elif int(conditional.predicate_value_id) in self.sequence_descriptors:
            predicate_sequence_id = int(conditional.predicate_value_id)
        if conditional.entry_record_projections:
            pass
        elif predicate_sequence_id is not None:
            descriptor = self.sequence_descriptors[predicate_sequence_id]
            length_cell = self.sequence_storage_values[
                predicate_sequence_id
            ][len(descriptor.column_value_ids)]
            length = self.fresh_value(dtype="int64")
            predicate = self.fresh_value(dtype="bool")
            self.emit(
                Handler.Load,
                [length_cell],
                length,
                attributes={
                    "binding": "resident_sequence_truthiness",
                    "sequence_id": predicate_sequence_id,
                },
            )
            self.emit(
                Handler.Gt,
                [length, self.constant_value(0)],
                predicate,
                attributes={
                    "binding": "resident_sequence_truthiness",
                    "sequence_id": predicate_sequence_id,
                },
            )
        elif int(conditional.predicate_value_id) in self.iterable_source_ids:
            predicate_source_id = int(conditional.predicate_value_id)
            length = self.expression_value(
                f"__iterable_extent_{predicate_source_id}__",
                location=f"{path}.truthiness",
            )
            predicate = self.fresh_value(dtype="bool")
            self.emit(
                Handler.Gt,
                [length, self.constant_value(0)],
                predicate,
                attributes={
                    "binding": "resident_iterable_truthiness",
                    "source_value_id": predicate_source_id,
                },
            )
        else:
            predicate = (
                self.lower_control_expression(
                    conditional.predicate_expression
                )
                if conditional.predicate_expression is not None
                else self.external_value(
                    conditional.predicate_value_id, dtype="bool"
                )
            )
        true_block = self.new_block("if_true")
        false_block = self.new_block("if_false")
        merge_block = self.new_block("if_merge")
        self.conditional_branch(
            predicate,
            true_block if conditional.expect_true else false_block,
            false_block if conditional.expect_true else true_block,
        )

        # Snapshot incoming scalar versions before lowering either arm.
        # Nested conditionals publish their merge by updating
        # ``external_values[initial_id]`` for the continuation.  Without this
        # boundary the child in the false arm can overwrite the parent's
        # incoming binding before the parent Phi is assembled, making both
        # parent inputs name the child result.  The deterministic SSA identity
        # history remains unchanged; this only preserves its path-specific
        # value object at the branch entry.
        carried_snapshots = {
            int(initial_value_id): self.external_value(int(initial_value_id))
            for _true_value_id, _false_value_id, initial_value_id,
            _merged_value_id in conditional.carried_aliases
        }

        def nested_conditional_aliases(block: ControlBlock):
            if isinstance(block, ConditionalBlock):
                yield from block.carried_aliases
                yield from nested_conditional_aliases(block.body)
                if block.orelse is not None:
                    yield from nested_conditional_aliases(block.orelse)
            elif isinstance(block, SequenceBlock):
                for child in block.blocks:
                    yield from nested_conditional_aliases(child)
            elif isinstance(block, (LoopBlock, WhileBlock)):
                yield from nested_conditional_aliases(block.body)
                for child in block.terminal_controls:
                    yield from nested_conditional_aliases(child)
            elif isinstance(block, CallBlock):
                yield from nested_conditional_aliases(block.callee)
            elif isinstance(block, ParallelDeployment):
                for lane in block.lanes:
                    yield from nested_conditional_aliases(lane)
            elif isinstance(block, StateMachineTick):
                for _value, child in block.cases:
                    yield from nested_conditional_aliases(child)
                if block.default is not None:
                    yield from nested_conditional_aliases(block.default)

        nested_aliases = tuple(nested_conditional_aliases(conditional.body))
        if conditional.orelse is not None:
            nested_aliases += tuple(nested_conditional_aliases(
                conditional.orelse
            ))
        nested_initial_ids = {
            int(initial_id)
            for _true_id, _false_id, initial_id, _merged_id
            in nested_aliases
        }
        nested_merge_ids = {
            int(merged_id)
            for _true_id, _false_id, _initial_id, merged_id
            in nested_aliases
        }
        for true_id, false_id, _initial_id, _merged_id in (
            conditional.carried_aliases
        ):
            for value_id in (int(true_id), int(false_id)):
                if (
                    value_id in nested_initial_ids
                    and value_id not in nested_merge_ids
                    and value_id not in carried_snapshots
                ):
                    carried_snapshots[value_id] = self.external_value(value_id)

        self.current = true_block
        for (
            origin_sequence_id, handle_id, result_id, column, dtype,
        ) in conditional.entry_record_projections:
            descriptor = self.sequence_descriptors.get(int(origin_sequence_id))
            if (
                descriptor is None
                or not 0 <= int(column) < len(descriptor.column_value_ids)
            ):
                self.shortfalls.append(SSALoweringShortfall(
                    "control",
                    "optional-record-projection",
                    path,
                    "optional record field has no resident origin column: "
                    f"sequence={origin_sequence_id} column={column}",
                ))
                continue
            source = self.external_value(
                int(descriptor.column_value_ids[int(column)]),
                dtype=str(dtype),
            )
            self.indexed_load(
                source,
                self.external_value(int(handle_id), dtype="int"),
                int(result_id),
                attributes={
                    "binding": "optional_record_field",
                    "origin_sequence_id": int(origin_sequence_id),
                    "column": int(column),
                },
                claim_provisional_definition=True,
            )
        self.lower(conditional.body, path=f"{path}.body")
        for true_id, _false_id, initial_id, merged_id in (
            conditional.carried_sequence_aliases
        ):
            if int(true_id) != int(initial_id):
                self.lower_sequence_mutation(
                    ControlSequenceMutation(
                        int(initial_id), "replace", (int(true_id),),
                        int(merged_id), policy="duplicates",
                    ),
                    path=f"{path}.body",
                )
        true_exit = self.current
        if not true_exit.successors:
            self.branch(merge_block)

        self.current = false_block
        if conditional.orelse is not None:
            self.lower(conditional.orelse, path=f"{path}.orelse")
        for _true_id, false_id, initial_id, merged_id in (
            conditional.carried_sequence_aliases
        ):
            if int(false_id) != int(initial_id):
                self.lower_sequence_mutation(
                    ControlSequenceMutation(
                        int(initial_id), "replace", (int(false_id),),
                        int(merged_id), policy="duplicates",
                    ),
                    path=f"{path}.orelse",
                )
        false_exit = self.current
        if not false_exit.successors:
            self.branch(merge_block)

        self.current = merge_block
        for (
            true_value_id, false_value_id, initial_value_id, merged_value_id,
        ) in conditional.carried_aliases:
            initial = carried_snapshots[int(initial_value_id)]
            true_value = (
                carried_snapshots.get(int(true_value_id), initial)
                if int(true_value_id) in carried_snapshots
                else self.external_value(int(true_value_id), dtype=initial.dtype)
            )
            false_value = (
                carried_snapshots.get(int(false_value_id), initial)
                if int(false_value_id) in carried_snapshots
                else self.external_value(int(false_value_id), dtype=initial.dtype)
            )
            merged = SSAValue(
                int(merged_value_id),
                dtype=initial.dtype,
                shape=initial.shape,
            )
            self.emit(
                Handler.Phi,
                [true_value, false_value],
                merged,
                attributes={
                    "incoming_blocks": (true_exit.name, false_exit.name),
                    "binding": "conditional_carried",
                    "initial_value_id": int(initial_value_id),
                },
            )
            self.external_values[int(merged_value_id)] = merged
            self.external_values[int(initial_value_id)] = merged
        for _true_id, _false_id, initial_id, merged_id in (
            conditional.carried_sequence_aliases
        ):
            resident = self.external_value(int(initial_id))
            self.external_values[int(merged_id)] = resident

    def _publish_loop_result_ports(
        self,
        loop: LoopBlock | WhileBlock,
        *,
        header: BasicBlock,
        exit_block: BasicBlock,
        carried: list[tuple[int, int, SSAValue, SSAValue, SSAValue]],
        break_edges: tuple[tuple[BasicBlock, tuple[SSAValue, ...]], ...],
    ) -> None:
        """Define authored LoopResult ids with edge-correct exit Phis."""

        carried_by_updated = {
            int(updated_id): (index, current)
            for index, (updated_id, _initial_id, _initial, _updated, current)
            in enumerate(carried)
        }
        carried_ports = getattr(self, "_carried_port_groups", None)
        if carried_ports is None:
            carried_ports = {}
            self._carried_port_groups = carried_ports
        port_values = getattr(self, "_carried_port_values", None)
        if port_values is None:
            port_values = {}
            self._carried_port_values = port_values

        self.current = exit_block
        for port_id, initial_id, updated_id in getattr(
            loop, "result_ports", ()
        ):
            carried_entry = carried_by_updated.get(int(updated_id))
            if carried_entry is None:
                continue
            carried_index, normal_value = carried_entry
            incoming_blocks = [header.name]
            incoming_values = [normal_value]
            for predecessor, edge_values in break_edges:
                incoming_blocks.append(predecessor.name)
                incoming_values.append(edge_values[carried_index])

            # A post-loop call may have exposed this authored identity as a
            # provisional formal while structural blocks were assembled.  Use
            # that exact SSAValue as the definition so already-built operands
            # become resident, then retire its provisional ABI role.
            port = self.produced_value(
                int(port_id),
                dtype=str(normal_value.dtype or "unknown"),
                claim_provisional_definition=True,
            )
            self.emit(
                Handler.Phi,
                incoming_values,
                port,
                attributes={
                    "incoming_blocks": tuple(incoming_blocks),
                    "binding": "loop_result_port",
                    "initial_value_id": int(initial_id),
                    "updated_value_id": int(updated_id),
                },
            )
            group = carried_ports.setdefault(
                (int(initial_id), int(updated_id)), set()
            )
            group.add(int(port_id))
            port.accounting["carried_port_ids"] = tuple(dict.fromkeys((
                *(port.accounting.get("carried_port_ids") or ()),
                *sorted(int(item) for item in group),
            )))
            # Nested loops can expose successive LoopResult spellings for
            # the same carried binding.  Once the enclosing loop exits, every
            # spelling in that equivalence class denotes the enclosing exit
            # value.  Rebind the lookup table (without rewriting operands
            # already emitted inside the loop) so post-loop expressions that
            # retained an inner port id cannot bypass the outer zero-trip
            # path and consume a non-dominating inner Phi.
            for equivalent_port_id in group:
                self.external_values[int(equivalent_port_id)] = port
                port_values[int(equivalent_port_id)] = port

    def lower_loop(self, loop: LoopBlock, *, path: str) -> None:
        recursion_region_id = loop.recursion_region_id
        deployment_id = None
        if loop.parallel_iterations:
            deployment_id = self.next_deployment_id
            self.next_deployment_id += 1
            record = {
                "region_id": deployment_id,
                "kind": "parallel_candidate",
                "schedule": "independent_iterations",
                "schedule_preference": loop.schedule_preference,
                "lane_count": 1,
                "iteration_space": (loop.start, loop.stop, loop.step),
                "carried_aliases": tuple(loop.carried_aliases),
                "recursion_region_id": recursion_region_id,
                "origin": "retained_loop",
                "source_loop_node_id": loop.source_loop_node_id,
                "scale": 1,
                "join": DeploymentJoin(),
                "declared_lanes": (),
            }
            self.deployment_records.append(record)
        preheader = self.current
        start = self.expression_value(
            loop.start,
            location=f"{path}.start",
        )
        stop = self.expression_value(
            loop.stop,
            location=f"{path}.stop",
        )
        step = self.expression_value(
            loop.step,
            location=f"{path}.step",
        )
        # A carried seed whose graph node was folded to a constant must enter
        # as that literal; external_value would otherwise invent a
        # producerless argument for it.
        for seed_id, seed_literal in getattr(loop, "carried_seeds", ()):
            seed_id = int(seed_id)
            if seed_id in self.external_values:
                continue
            seed_value = self.fresh_value(dtype="float64")
            self.emit(
                Handler.Const, [], seed_value,
                attributes={"value": float(seed_literal)},
            )
            self.external_values[seed_id] = seed_value
        carried: list[tuple[int, int, SSAValue, SSAValue, SSAValue]] = []
        for updated_id, initial_id in loop.carried_aliases:
            updated_id = int(updated_id)
            initial_id = int(initial_id)
            initial_value = self.external_value(initial_id)
            updated_value = SSAValue(
                updated_id,
                dtype=initial_value.dtype,
                shape=initial_value.shape,
            )
            current_value = self.fresh_value(
                dtype=initial_value.dtype,
                shape=initial_value.shape,
            )
            current_value.accounting.update({
                "source_value_id": initial_id,
                "carried_from_value_id": updated_id,
            })
            # Region output extraction will use this exact object as the
            # backedge definition referenced by the Phi below.
            self.external_values[updated_id] = updated_value
            carried.append(
                (
                    updated_id,
                    initial_id,
                    initial_value,
                    updated_value,
                    current_value,
                )
            )
        header = self.new_block("loop_header")
        body = self.new_block("loop_body")
        latch = self.new_block("loop_latch")
        exit_block = self.new_block("loop_exit")
        self.current = preheader
        # Seed every carried SLOT from its initial value before entering the
        # loop. The carried updated id is one storage slot: the body reads it
        # at the top of an iteration (a linked call's argument binds there --
        # that is the aliased-input/output design, one slot per carried value,
        # no copies inside the loop) and writes it before the latch. Reading
        # slot-then-writing-slot is correct on every iteration EXCEPT the
        # first, where nothing has written the slot yet; without this seed the
        # first iteration reads an uninitialized slot, which the materializer
        # reports as use-before-def and a native backend reads as garbage.
        # The seed is emitted onto the SAME reserved SSAValue object the body
        # will later redefine -- in-place reuse of one slot, which the
        # well-formedness rules accept for one object (diagnose stage 2a).
        for _updated_id, _initial_id, initial_value, reserved, _current in carried:
            self.emit(
                Handler.Cast,
                [initial_value],
                reserved,
                attributes={
                    "binding": "loop_carried_seed",
                    "target_dtype": str(initial_value.dtype or "float64"),
                },
            )
        if deployment_id is not None:
            self.emit_deployment_boundary(Handler.Deploy, record)
        self.branch(header)

        induction = self.fresh_value(dtype="int")
        next_induction = self.fresh_value(dtype="int")
        self.current = header
        self.emit(
            Handler.Phi,
            [start, next_induction],
            induction,
            attributes={
                "incoming_blocks": (preheader.name, latch.name),
                "source_name": loop.induction,
                "recursion_region_id": recursion_region_id,
                "source_loop_node_id": loop.source_loop_node_id,
            },
        )
        carried_phis: dict[int, Instr] = {}
        for (
            updated_id,
            initial_id,
            initial_value,
            updated_value,
            current_value,
        ) in carried:
            self.emit(
                Handler.Phi,
                [initial_value, updated_value],
                current_value,
                attributes={
                    "incoming_blocks": (preheader.name, latch.name),
                    "binding": "loop_carried",
                    "initial_value_id": initial_id,
                    "updated_value_id": updated_id,
                    "recursion_region_id": recursion_region_id,
                },
            )
            carried_phis[updated_id] = self.current.instrs[-1]
            self.external_values[initial_id] = current_value
        condition = self.fresh_value(dtype="bool")
        self.emit(
            Handler.Lt if loop.comparison == "lt" else Handler.Gt,
            [induction, stop],
            condition,
            attributes={"binding": "loop_condition", "comparison": loop.comparison},
        )
        self.conditional_branch(condition, body, exit_block)

        self.current = body
        self.loop_targets.append((latch, exit_block))
        exit_context = {
            "carried": tuple(
                (updated_id, initial_id, current)
                for updated_id, initial_id, _initial, _updated, current
                in carried
            ),
            "break_edges": [],
        }
        self.loop_exit_contexts.append(exit_context)
        if deployment_id is not None:
            # Lane zero is the SSA template for every independent iteration;
            # the iteration space above tells a deployment pass how to fan it
            # out.  The ordinary CFG remains the serial fallback.
            self.active_deployments.append((deployment_id, 0))
        previous_induction = self.local_control_values.get(loop.induction)
        self.local_control_values[loop.induction] = induction
        restored_values: dict[int, SSAValue | None] = {}
        bound_loop_target_ids: set[int] = set()
        for iterable_id, target_id, induction_name in (
            self.program.iterable_bindings
        ):
            if induction_name != loop.induction:
                continue
            bound_loop_target_ids.add(int(target_id))
            restored_values[int(target_id)] = self.external_values.get(
                int(target_id)
            )
            child_selection = self.child_table_selections.get(int(iterable_id))
            if child_selection is not None:
                pool, handle = child_selection
                base_offset = self.fresh_value(dtype="int")
                offset = self.fresh_value(dtype="int")
                address = self.fresh_value(dtype="ptr")
                target = self.produced_value(int(target_id), dtype="unknown")
                self.emit(
                    Handler.Mul,
                    [handle, self.external_value(pool.row_stride_value_id)],
                    base_offset,
                    attributes={"binding": "child_table_offset"},
                )
                self.emit(
                    Handler.Add, [base_offset, induction], offset,
                    attributes={"binding": "child_table_offset"},
                )
                if pool.live_flags_value_id is not None:
                    live_address = self.fresh_value(dtype="ptr")
                    live = self.fresh_value(dtype="bool")
                    active = self.new_block("child_table_live")
                    self.emit(
                        Handler.GetElementPtr,
                        [self.external_value(pool.live_flags_value_id), offset],
                        live_address,
                        attributes={"binding": "child_table_live"},
                    )
                    self.emit(
                        Handler.Load, [live_address], live,
                        attributes={"binding": "child_table_live"},
                    )
                    self.conditional_branch(live, active, latch)
                    self.current = active
                self.emit(
                    Handler.GetElementPtr,
                    [self.external_value(pool.column_value_ids[0]), offset],
                    address,
                    attributes={"binding": "child_table_key"},
                )
                self.emit(
                    Handler.Load, [address], target,
                    attributes={"binding": "child_table_key"},
                )
                continue
            if int(target_id) in self.nested_row_target_ids:
                self.bind_nested_row(
                    self.external_value(iterable_id, dtype="int"),
                    induction,
                    int(target_id),
                    child_key=("iterable", int(iterable_id), -1),
                    attributes={"induction": loop.induction},
                )
                continue
            self.indexed_load(
                self.external_value(iterable_id),
                induction,
                target_id,
                attributes={
                    "binding": "iterable",
                    "induction": loop.induction,
                },
            )
        for iterable_id, target_id, induction_name, projection in (
            self.program.projected_iterable_bindings
        ):
            if induction_name != loop.induction:
                continue
            bound_loop_target_ids.add(int(target_id))
            restored_values[int(target_id)] = self.external_values.get(
                int(target_id)
            )
            if projection == "induction":
                self.external_values[int(target_id)] = induction
                continue
            projection_index = induction
            column_projection = projection
            if (
                isinstance(projection, tuple)
                and len(projection) == 3
                and projection[0] == "column_at_value"
            ):
                column_projection = int(projection[1])
                projection_index = self.external_value(
                    int(projection[2]), dtype="int"
                )
            source = self.external_value(iterable_id)
            if projection is not None:
                column_key = (int(iterable_id), int(column_projection))
                source = self.projected_row_columns.get(column_key)
                if source is None:
                    target_meta = self.region_value_meta.get(int(target_id))
                    target_dtype = (
                        None if target_meta is None else str(target_meta.dtype)
                    )
                    if (
                        int(target_id) in self.nested_row_target_ids
                        and int(target_id)
                        not in self.variant_projected_target_ids
                    ):
                        target_dtype = "int"
                    resident_sequence_id = int(iterable_id)
                    seen_aliases: set[int] = set()
                    while (
                        resident_sequence_id in self.value_aliases
                        and resident_sequence_id not in seen_aliases
                    ):
                        seen_aliases.add(resident_sequence_id)
                        resident_sequence_id = int(
                            self.value_aliases[resident_sequence_id]
                        )
                    descriptor = self.sequence_descriptors.get(
                        resident_sequence_id
                    )
                    if (
                        descriptor is not None
                        and 0 <= int(column_projection) < len(
                            descriptor.column_value_ids
                        )
                    ):
                        source = self.external_value(
                            int(descriptor.column_value_ids[int(column_projection)]),
                            dtype=str(descriptor.column_dtypes[int(column_projection)]),
                        )
                    elif int(column_projection) == 0:
                        source = self.external_value(
                            iterable_id, dtype=target_dtype
                        )
                        source.accounting.update({
                            "projected_row_source_id": int(iterable_id),
                            "projected_row_column": int(column_projection),
                        })
                    else:
                        source = self.fresh_value(dtype=target_dtype)
                        source.accounting.update({
                            "projected_row_source_id": int(iterable_id),
                            "projected_row_column": int(column_projection),
                        })
                        self.arguments.append(source)
                    self.projected_row_columns[column_key] = source
                if int(target_id) in self.variant_projected_target_ids:
                    scalar_source = source
                    handle_source = self.variant_handle_columns.get(column_key)
                    if handle_source is None:
                        handle_source = self.fresh_value(dtype="int")
                        handle_source.accounting.update({
                            "projected_variant_source_id": int(iterable_id),
                            "projected_variant_column": int(column_projection),
                        })
                        self.arguments.append(handle_source)
                        self.variant_handle_columns[column_key] = handle_source
                    scalar_value = self.indexed_load(
                        scalar_source,
                        projection_index,
                        target_id,
                        attributes={
                            "binding": "projected_variant_scalar",
                            "induction": loop.induction,
                            "projection": projection,
                        },
                    )
                    self.bind_nested_row(
                        handle_source,
                        projection_index,
                        int(target_id),
                        child_key=(
                            "projected", int(iterable_id),
                            int(column_projection)
                        ),
                        attributes={
                            "induction": loop.induction,
                            "projection": projection,
                            "variant": True,
                        },
                    )
                    self.variant_row_values[int(target_id)] = (
                        self.external_values[int(target_id)]
                    )
                    # Scalar regions continue to consume the scalar column.
                    self.external_values[int(target_id)] = scalar_value
                    continue
            if (
                projection is not None
                and int(target_id) in self.nested_row_target_ids
            ):
                self.bind_nested_row(
                    source,
                    projection_index,
                    int(target_id),
                    child_key=(
                        "projected", int(iterable_id), int(column_projection)
                    ),
                    attributes={
                        "induction": loop.induction,
                        "projection": projection,
                    },
                )
                continue
            self.indexed_load(
                source,
                projection_index,
                target_id,
                attributes={
                    "binding": "projected_iterable",
                    "iterable_id": int(iterable_id),
                    "induction": loop.induction,
                    "projection": projection,
                },
                claim_provisional_definition=True,
            )
        for base_id, column, result_id, induction_name in (
            self.nested_row_projections
        ):
            if induction_name != loop.induction:
                continue
            base = self.external_values.get(int(base_id))
            if base is None:
                continue
            column_value = self.fresh_value(dtype="int")
            self.emit(
                Handler.Const, [], column_value,
                attributes={"value": int(column)},
            )
            self.indexed_load(
                base,
                column_value,
                int(result_id),
                attributes={
                    "binding": "nested_row_projection",
                    "induction": loop.induction,
                    "projection": int(column),
                    "source_value_id": int(base_id),
                },
            )
        for iterable_id, target_id, induction_name, values in (
            self.program.static_iterable_bindings
        ):
            if induction_name != loop.induction:
                continue
            bound_loop_target_ids.add(int(target_id))
            restored_values[int(target_id)] = self.external_values.get(
                int(target_id)
            )
            aggregate = self.fresh_value(dtype="ssa.aggregate")
            self.emit(
                Handler.Const,
                [],
                aggregate,
                attributes={
                    "value": tuple(values),
                    "binding": "static_iterable",
                    "source_value_id": int(iterable_id),
                },
            )
            self.indexed_load(
                aggregate,
                induction,
                target_id,
                attributes={
                    "binding": "static_iterable",
                    "induction": loop.induction,
                },
            )
        for aggregate_id, target_id, induction_name, source_ids in (
            self.program.closure_iterable_bindings
        ):
            if induction_name != loop.induction:
                continue
            if int(target_id) in bound_loop_target_ids:
                # Closure captures are the fallback resident spelling for a
                # loop target.  If the same target already has an explicit
                # iterable/projected/static binding, emitting the fallback as
                # well defines one SSA identity twice and may replace a real
                # array load with a synthetic aggregate expression.
                continue
            bound_loop_target_ids.add(int(target_id))
            restored_values[int(target_id)] = self.external_values.get(
                int(target_id)
            )
            aggregate = self.fresh_value(dtype="ssa.aggregate")
            self.emit(
                Handler.Const,
                [self.external_value(value_id) for value_id in source_ids],
                aggregate,
                attributes={
                    "binding": "closure_iterable",
                    "source_value_id": int(aggregate_id),
                    "resident_source_ids": tuple(source_ids),
                },
            )
            self.indexed_load(
                aggregate,
                induction,
                target_id,
                attributes={
                    "binding": "closure_iterable",
                    "induction": loop.induction,
                },
            )
        blocks_before_body = {id(existing) for existing in self.blocks.values()}
        try:
            self.lower(loop.body, path=f"{path}.body")
            for mutation in loop.sequence_mutations:
                self.lower_sequence_mutation(mutation, path=path)
            for terminal in loop.terminal_controls:
                self.lower(terminal, path=f"{path}.terminal")
        finally:
            self.loop_exit_contexts.pop()
            self.loop_targets.pop()
        body_blocks = [
            candidate
            for candidate in self.blocks.values()
            if id(candidate) not in blocks_before_body or candidate is body
        ]
        # A nested retained loop may carry the same source state as this loop.
        # Its exit Phi is then the value produced by this iteration, but the
        # nested lowering necessarily replaced ``external_values[updated_id]``
        # with that inner value.  Point the enclosing Phi's latch operand at
        # the value the body actually published instead of requiring the
        # placeholder object reserved before the body was lowered.
        carried_updates: dict[int, SSAValue] = {}
        for updated_id, _initial_id, _initial, reserved, _current in carried:
            published = self.external_values.get(updated_id, reserved)
            carried_updates[updated_id] = published
            if published is not reserved:
                carried_phis[updated_id].args[1] = published
        # Producers are counted in the BODY's blocks only, as the shortfall
        # message has always claimed. Scanning every block would let the
        # preheader seed above stand in for a producer, silently accepting a
        # loop whose body updates nothing -- the carried value would freeze at
        # its seed, which is the exact miscompilation shape this check exists
        # to refuse.
        produced_results = {
            id(instruction.res)
            for basic_block in body_blocks
            for instruction in basic_block.instrs
            if instruction.res is not None
        }
        for updated_id, _initial_id, _initial, updated, _current in carried:
            if id(carried_updates[updated_id]) not in produced_results:
                self.shortfalls.append(
                    SSALoweringShortfall(
                        "control",
                        "loop_carried",
                        f"{path}.body",
                        f"carried update value {updated_id} has no producer "
                        "inside the loop body; "
                        f"alias_source={self.value_aliases.get(updated_id)!r}; "
                        "declared_region_outputs={}".format(tuple(
                            region_index
                            for region_index, (_feeds, outputs)
                            in self.region_signatures.items()
                            if updated_id in outputs
                        )),
                    )
                )
        for source_id, collection_id, induction_name, start in (
            self.program.collection_bindings
        ):
            if induction_name != loop.induction:
                continue
            publication_index = induction
            if int(start):
                offset = self.constant_value(int(start))
                publication_index = self.fresh_value(dtype="int")
                self.emit(
                    Handler.Add,
                    [induction, offset],
                    publication_index,
                    attributes={"binding": "collection_offset"},
                )
            address = self.fresh_value(dtype="ptr")
            self.emit(
                Handler.GetElementPtr,
                [
                    self.external_value(collection_id),
                    publication_index,
                ],
                address,
                attributes={
                    "binding": "collection_publication",
                    "collection_value_id": int(collection_id),
                    "induction": loop.induction,
                },
            )
            self.emit(
                Handler.Store,
                [self.external_value(source_id), address],
                attributes={
                    "binding": "collection_publication",
                    "source_value_id": int(source_id),
                },
            )
        if deployment_id is not None:
            self.active_deployments.pop()
        if not self.current.successors:
            self.branch(latch)

        self.current = latch
        self.emit(
            Handler.Add,
            [induction, step],
            next_induction,
        )
        self.branch(header)
        if recursion_region_id is not None:
            source = self.recursion_regions[int(recursion_region_id)]
            lowered_region = self.ssa_recursion_table.setdefault(
                int(recursion_region_id),
                {**source, "function": self.function_name, "loops": []},
            )
            lowered_region["loops"].append({
                "function": self.function_name,
                "preheader": preheader.name,
                "header": header.name,
                "body": body.name,
                "latch": latch.name,
                "exit": exit_block.name,
                "phi_value_ids": (
                    int(induction.id),
                    *(int(current.id) for *_prefix, current in carried),
                ),
                "backedge": (latch.name, header.name),
            })
        for target_id, previous in restored_values.items():
            if previous is None:
                self.external_values.pop(target_id, None)
            else:
                self.external_values[target_id] = previous
        if previous_induction is None:
            self.local_control_values.pop(loop.induction, None)
        else:
            self.local_control_values[loop.induction] = previous_induction
        for updated_id, initial_id, _initial, _updated, current in carried:
            self.external_values[initial_id] = current
            self.external_values[updated_id] = current
        self._publish_loop_result_ports(
            loop,
            header=header,
            exit_block=exit_block,
            carried=carried,
            break_edges=tuple(exit_context["break_edges"]),
        )
        if deployment_id is not None:
            self.emit_deployment_boundary(Handler.Join, record)

    def lower_while(self, loop: WhileBlock, *, path: str) -> None:
        recursion_region_id = loop.recursion_region_id
        self.lower(loop.condition, path=f"{path}.condition.initial")
        preheader = self.current
        initial_predicate = (
            self.lower_control_expression(loop.predicate_expression)
            if loop.predicate_expression is not None
            else self.external_value(loop.predicate_value_id, dtype="bool")
        )
        next_predicate = self.fresh_value(dtype="bool")

        # A carried seed whose graph node was folded to a constant must enter
        # as that literal; external_value would otherwise invent a
        # producerless argument for it.
        for seed_id, seed_literal in getattr(loop, "carried_seeds", ()):
            seed_id = int(seed_id)
            if seed_id in self.external_values:
                continue
            seed_value = self.fresh_value(dtype="float64")
            self.emit(
                Handler.Const, [], seed_value,
                attributes={"value": float(seed_literal)},
            )
            self.external_values[seed_id] = seed_value
        carried: list[tuple[int, int, SSAValue, SSAValue, SSAValue]] = []
        for updated_id, initial_id in loop.carried_aliases:
            updated_id = int(updated_id)
            initial_id = int(initial_id)
            initial_value = self.external_value(initial_id)
            updated_value = SSAValue(
                updated_id,
                dtype=initial_value.dtype,
                shape=initial_value.shape,
            )
            current_value = self.fresh_value(
                dtype=initial_value.dtype,
                shape=initial_value.shape,
            )
            self.external_values[updated_id] = updated_value
            carried.append((
                updated_id, initial_id, initial_value,
                updated_value, current_value,
            ))

        header = self.new_block("while_header")
        body = self.new_block("while_body")
        latch = self.new_block("while_latch")
        exit_block = self.new_block("while_exit")
        self.current = preheader
        self.branch(header)

        self.current = header
        current_predicate = self.fresh_value(dtype="bool")
        self.emit(
            Handler.Phi,
            [initial_predicate, next_predicate],
            current_predicate,
            attributes={
                "incoming_blocks": (preheader.name, latch.name),
                "binding": "while_condition",
                "source_value_id": int(loop.predicate_value_id),
                "recursion_region_id": recursion_region_id,
                "source_loop_node_id": loop.source_loop_node_id,
            },
        )
        self.external_values[int(loop.predicate_value_id)] = current_predicate
        carried_phis: dict[int, Instr] = {}
        for updated_id, initial_id, initial, updated, current in carried:
            self.emit(
                Handler.Phi,
                [initial, updated],
                current,
                attributes={
                    "incoming_blocks": (preheader.name, latch.name),
                    "binding": "loop_carried",
                    "initial_value_id": initial_id,
                    "updated_value_id": updated_id,
                    "recursion_region_id": recursion_region_id,
                },
            )
            carried_phis[updated_id] = self.current.instrs[-1]
            self.external_values[initial_id] = current
        self.conditional_branch(current_predicate, body, exit_block)

        self.current = body
        self.loop_targets.append((latch, exit_block))
        exit_context = {
            "carried": tuple(
                (updated_id, initial_id, current)
                for updated_id, initial_id, _initial, _updated, current
                in carried
            ),
            "break_edges": [],
        }
        self.loop_exit_contexts.append(exit_context)
        preserved_before_body = set(self.preserved_region_output_ids)
        self.preserved_region_output_ids.update(
            int(initial_id) for _updated_id, initial_id, *_rest in carried
        )
        blocks_before_body = {id(existing) for existing in self.blocks.values()}
        try:
            self.lower(loop.body, path=f"{path}.body")
            for mutation in loop.sequence_mutations:
                self.lower_sequence_mutation(mutation, path=path)
            for terminal in loop.terminal_controls:
                self.lower(terminal, path=f"{path}.terminal")
        finally:
            self.preserved_region_output_ids = preserved_before_body
            self.loop_exit_contexts.pop()
            self.loop_targets.pop()
        body_blocks = [
            candidate
            for candidate in self.blocks.values()
            if id(candidate) not in blocks_before_body or candidate is body
        ]
        carried_updates: dict[int, SSAValue] = {}
        for updated_id, _initial_id, _initial, reserved, _current in carried:
            published = self.external_values.get(updated_id, reserved)
            carried_updates[updated_id] = published
            if published is not reserved:
                carried_phis[updated_id].args[1] = published
        produced_results = {
            id(instruction.res)
            for basic_block in body_blocks
            for instruction in basic_block.instrs
            if instruction.res is not None
        }
        for updated_id, _initial_id, _initial, _updated, current in carried:
            if id(carried_updates[updated_id]) not in produced_results:
                declared_outputs = tuple(
                    region_index
                    for region_index, (_feeds, outputs)
                    in self.region_signatures.items()
                    if updated_id in outputs
                )
                if not declared_outputs:
                    # A syntactic identity assignment (``value = value``) or
                    # an untouched arm of a partial assignment has no body
                    # instruction to publish. Its latch definition is the
                    # current iteration's header Phi, not the producerless
                    # placeholder reserved for a possible real update.
                    carried_phis[updated_id].args[1] = current
                    carried_updates[updated_id] = current
                    self.external_values[updated_id] = current
                    current.accounting["ssa_identity_backedge"] = True
                    continue
                self.shortfalls.append(SSALoweringShortfall(
                    "control",
                    "loop_carried",
                    f"{path}.body",
                    f"carried update value {updated_id} has no producer "
                    "inside the while-loop body; "
                    f"alias_source={self.value_aliases.get(updated_id)!r}; "
                    f"declared_region_outputs={declared_outputs}",
                ))
        if not self.current.successors:
            self.branch(latch)

        self.current = latch
        # The latch evaluates the guard for the NEXT iteration, so every
        # carried name must resolve to the body's UPDATED value there --
        # not the header phi it is bound to for the rest of the loop.
        # Testing the pre-update value lags the guard one iteration behind
        # and runs the loop once too often (scorecard level 17: the
        # compiled while halved 1.0 to 0.5 where the authored loop stops).
        latch_restore = []
        for updated_id, initial_id, _initial, updated, _current in carried:
            latch_restore.append(
                (initial_id, self.external_values.get(initial_id))
            )
            self.external_values[initial_id] = self.external_values.get(
                updated_id, updated
            )
        self.external_values[int(loop.predicate_value_id)] = next_predicate
        preserved_before = set(self.preserved_region_output_ids)
        self.preserved_region_output_ids.update(
            int(initial_id) for _updated_id, initial_id, *_rest in carried
        )
        try:
            self.lower(loop.condition, path=f"{path}.condition.latch")
        finally:
            self.preserved_region_output_ids = preserved_before
        if loop.predicate_expression is not None:
            self.lower_control_expression(
                loop.predicate_expression,
                result_override=next_predicate,
            )
        # Post-loop consumers read the converged header phi, not the last
        # body update: restore the loop-wide binding before leaving.
        for initial_id, previous in latch_restore:
            if previous is None:
                self.external_values.pop(initial_id, None)
            else:
                self.external_values[initial_id] = previous
        if not any(
            instruction.res is next_predicate
            for instruction in self.current.instrs
        ):
            self.shortfalls.append(SSALoweringShortfall(
                "control",
                "while_condition",
                f"{path}.condition",
                f"predicate value {loop.predicate_value_id} has no producer",
            ))
        if not self.current.successors:
            self.branch(header)

        if recursion_region_id is not None:
            source = self.recursion_regions[int(recursion_region_id)]
            lowered = self.ssa_recursion_table.setdefault(
                int(recursion_region_id),
                {**source, "function": self.function_name, "loops": []},
            )
            lowered["loops"].append({
                "function": self.function_name,
                "header": header.name,
                "body": body.name,
                "latch": latch.name,
                "exit": exit_block.name,
                "phi_value_ids": (
                    int(current_predicate.id),
                    *(int(current.id) for *_prefix, current in carried),
                ),
                "backedge": (latch.name, header.name),
                "domain": "condition",
                "source_loop_node_id": loop.source_loop_node_id,
            })
        for updated_id, initial_id, _initial, _updated, current in carried:
            self.external_values[initial_id] = current
            self.external_values[updated_id] = current
        self.external_values[int(loop.predicate_value_id)] = current_predicate
        self._publish_loop_result_ports(
            loop,
            header=header,
            exit_block=exit_block,
            carried=carried,
            break_edges=tuple(exit_context["break_edges"]),
        )

    def lower_state_machine(
        self,
        tick: StateMachineTick,
        *,
        path: str,
    ) -> None:
        state = self.uniform_values.get(str(tick.state))
        if state is None:
            state = self.expression_value(
                tick.state,
                location=f"{path}.state",
            )
        merge = self.new_block("state_merge")
        for index, (case_value, case_body) in enumerate(tick.cases):
            case = self.new_block("state_case")
            otherwise = self.new_block("state_next")
            literal = self.expression_value(
                case_value,
                location=f"{path}.case[{index}]",
            )
            condition = self.fresh_value(dtype="bool")
            self.emit(Handler.Eq, [state, literal], condition)
            self.conditional_branch(condition, case, otherwise)
            self.current = case
            self.lower(case_body, path=f"{path}.case[{index}].body")
            if not self.current.successors:
                self.branch(merge)
            self.current = otherwise
        if tick.default is not None:
            self.lower(tick.default, path=f"{path}.default")
        if not self.current.successors:
            self.branch(merge)
        self.current = merge

    def finish(self) -> tuple[Function, tuple[SSALoweringShortfall, ...]]:
        returned = []
        named_returns = []
        returned_ids = set()
        # Every identity a RESOLVED named return has ever worn. An output
        # id that is an earlier binding of such a name is the same authored
        # output seen through a stale identity -- a while loop rebinding
        # ``total`` leaves the return node's captured id pointing at the
        # pre-loop value, and returning both publishes the scalar twice
        # (scorecard level 17: (0.5, 0.5) for 0.5).
        superseded: set[int] = set()
        carried_port_values = getattr(self, "_carried_port_values", {})

        def existing_value(value_id: int) -> SSAValue | None:
            """Resolve an authored assignment alias without inventing an ABI input.

            An alias chain such as ``surface_velocity[..., 1] = ...`` over
            ``surface_velocity[..., 2] = ...`` over ``point * 0.0`` names ONE
            storage in successive versions.  The region that performs the
            writes publishes the version it produced (the latest), not the
            root arena, so the published value is found by checking each
            version from the returned identity toward the root and taking the
            first one that exists.  Looking up only the root dropped every
            return built by in-place stores, and for an in/out parameter
            returned the pre-write arena that the stale-identity filter then
            removed.
            """
            value_id = int(value_id)
            seen: set[int] = set()
            while True:
                found = self.external_values.get(value_id)
                if found is not None:
                    return found
                if value_id in self.value_aliases and value_id not in seen:
                    seen.add(value_id)
                    value_id = int(self.value_aliases[value_id])
                    continue
                return None

        for name, history in self.named_output_histories.items():
            value = next((
                resolved
                for value_id in reversed(history)
                for resolved in (existing_value(value_id),)
                if resolved is not None
            ), None)
            if value is None:
                continue
            # A returned name whose final identity is a LoopResult port means
            # the carried phi -- the port id doubles as a written field slot
            # whose cell nothing stores.
            value = carried_port_values.get(int(value.id), value)
            named_returns.append((name, int(value.id)))
            superseded.update(map(int, history))
            if value.id not in returned_ids:
                returned.append(value)
                returned_ids.add(value.id)
        for value_id in self.output_value_ids:
            value = self.external_values.get(value_id)
            if value is not None:
                value = carried_port_values.get(int(value.id), value)
            if value is None or value.id in returned_ids:
                continue
            if int(value_id) in superseded or int(value.id) in superseded:
                continue
            returned.append(value)
            returned_ids.add(value.id)
        value_names = []
        for name, history in self.value_name_histories.items():
            value = next((
                self.external_values[value_id]
                for value_id in reversed(history)
                if value_id in self.external_values
            ), None)
            if (
                value is not None
                and int(value.id) not in self.declared_parameter_only_ids
            ):
                value_names.append((name, int(value.id)))
        argument_ids = {int(value.id) for value in self.arguments}
        parameter_value_names = tuple(
            (name, value_id)
            for name, value_id in value_names
            if name in self.parameter_names and int(value_id) in argument_ids
        )
        # Structured control expressions can consume the initial authored
        # parameter and then publish a loop-carried alias as the latest value
        # in its identity history.  Keep the actual ABI argument named when it
        # is referenced by emitted SSA; do not revive parameters that were
        # preserved in the signature but genuinely unused.
        used_value_ids = {
            int(value.id)
            for block in self.blocks.values()
            for instruction in block.instrs
            for value in instruction.args
        }
        named_parameters = {name for name, _ in parameter_value_names}
        parameter_value_names += tuple(
            (name, value_id)
            for name in self.parameter_names
            if name not in named_parameters
            for value_id in self.value_name_histories.get(name, ())
            if int(value_id) in argument_ids and int(value_id) in used_value_ids
        )
        deployment_regions = []
        for record in self.deployment_records:
            declared_lanes = {
                int(lane.index): lane
                for lane in record.get("declared_lanes", ())
            }
            lane_sites: dict[int, list[tuple[str, int]]] = {
                index: [] for index in range(int(record["lane_count"]))
            }
            lane_callees: dict[int, list[str]] = {
                index: [] for index in lane_sites
            }
            lane_regions: dict[int, list[int]] = {
                index: [] for index in lane_sites
            }
            for block_name, basic_block in self.blocks.items():
                for instruction_index, instruction in enumerate(
                    basic_block.instrs
                ):
                    memberships = tuple(
                        instruction.attributes.get(
                            "deployment_memberships", ()
                        )
                    )
                    for region_id, lane_index in memberships:
                        if int(region_id) != int(record["region_id"]):
                            continue
                        lane_index = int(lane_index)
                        lane_sites[lane_index].append(
                            (block_name, instruction_index)
                        )
                        callee = instruction.attributes.get("callee")
                        if callee is not None:
                            lane_callees[lane_index].append(str(callee))
                        source_region = instruction.attributes.get(
                            "region_index"
                        )
                        if source_region is not None:
                            lane_regions[lane_index].append(
                                int(source_region)
                            )
            deployment_regions.append(SSADeploymentRegion(
                region_id=int(record["region_id"]),
                function=self.function_name,
                kind=str(record["kind"]),
                schedule=str(record["schedule"]),
                schedule_preference=str(record["schedule_preference"]),
                lanes=tuple(
                    SSADeploymentLane(
                        index=index,
                        instruction_sites=tuple(lane_sites[index]),
                        callees=tuple(dict.fromkeys(lane_callees[index])),
                        source_region_indices=tuple(dict.fromkeys(
                            lane_regions[index]
                        )),
                        source_value_ids=tuple(
                            declared_lanes[index].value_ids
                            if index in declared_lanes else ()
                        ),
                        source_node_ids=tuple(
                            declared_lanes[index].source_node_ids
                            if index in declared_lanes else ()
                        ),
                    )
                    for index in sorted(lane_sites)
                ),
                iteration_space=record["iteration_space"],
                carried_aliases=tuple(record["carried_aliases"]),
                recursion_region_id=record["recursion_region_id"],
                origin=str(record["origin"]),
                source_loop_node_id=record["source_loop_node_id"],
                scale=int(record.get("scale", 1)),
                join=record["join"],
                deploy_site=record.get("deploy_site"),
                join_site=record.get("join_site"),
            ))
        if not self.current.successors and (
            not self.current.instrs
            or self.current.instrs[-1].op
            not in {Handler.Br.value, Handler.CondBr.value, Handler.Ret.value}
        ):
            self.emit(Handler.Ret, returned)
        function = Function(
                self.function_name,
                self.arguments,
                self.blocks,
                metadata={
                    "recursion_table": dict(self.ssa_recursion_table),
                    "named_outputs": tuple(named_returns),
                    # port id -> the carried phi VALUE standing at that
                    # port after the loops.  The record-return expansion
                    # resolves layout components by id, and a component
                    # whose id doubles as a written field slot must resolve
                    # to the phi, not the unwritten slot argument.
                    "carried_port_values": dict(
                        getattr(self, "_carried_port_values", {}) or {}
                    ),
                    "value_aliases": dict(self.value_aliases),
                    "value_names": tuple(value_names),
                    "parameter_names": parameter_value_names,
                    "validation_contracts": tuple(self.validation_contracts),
                    "control_ir": True,
                    "deployment_regions": tuple(deployment_regions),
                    "sequence_table": SSASequenceTable(
                        dict(self.sequence_descriptors)
                    ),
                    "sequence_helper_functions": tuple(
                        self.sequence_helper_functions.values()
                    ),
                    "sequence_array_argument_ids": tuple(dict.fromkeys(
                        [
                            int(value.id)
                            for value in self.projected_row_columns.values()
                        ] + [
                            int(value.id)
                            for value in self.variant_row_values.values()
                        ] + [
                        int(value_id)
                        for descriptor in self.sequence_descriptors.values()
                        for value_id in (
                            *descriptor.column_value_ids,
                            descriptor.length_address_id,
                            *(
                                (descriptor.status_address_id,)
                                if descriptor.status_address_id is not None else ()
                            ),
                            *(
                                (descriptor.live_flags_value_id,)
                                if descriptor.live_flags_value_id is not None else ()
                            ),
                            *(
                                (
                                    *descriptor.child_table_pool.column_value_ids,
                                    descriptor.child_table_pool.length_value_id,
                                    *((descriptor.child_table_pool.status_value_id,)
                                      if descriptor.child_table_pool.status_value_id is not None else ()),
                                    *((descriptor.child_table_pool.live_flags_value_id,)
                                      if descriptor.child_table_pool.live_flags_value_id is not None else ()),
                                )
                                if descriptor.child_table_pool is not None else ()
                            ),
                        )]
                    )),
                    "scalar_variant_argument_ids": tuple(sorted(
                        self.variant_projected_target_ids
                    )),
                    "projected_row_tables": tuple(
                        (
                            int(source_id), int(projection), int(value.id),
                            str(value.dtype or "unknown"),
                        )
                        for (source_id, projection), value
                        in self.projected_row_columns.items()
                    ),
                    "nested_child_tables": tuple(
                        (
                            str(source_kind), int(source_id), int(projection),
                            int(data.id), int(lengths.id),
                            int(sequence_stride.id), int(row_stride.id),
                        )
                        for (source_kind, source_id, projection), (
                            data, lengths, sequence_stride, row_stride
                        )
                        in self.nested_child_rows.items()
                    ),
                    "joined_sequence_views": tuple(
                        (int(outer_id), int(flat_id))
                        for outer_id, flat_id
                        in sorted(self.joined_flat_sequence_ids.items())
                    ),
                    # Exact authored identities for the resident sequence
                    # arenas.  Several names may intentionally alias one
                    # sequence id; retain every occurrence in source-table
                    # order instead of choosing a convenient spelling.
                    "sequence_value_names": tuple(
                        (
                            int(sequence_id),
                            tuple(
                                name
                                for name, history in self.value_name_histories.items()
                                if int(sequence_id) in history
                            ),
                        )
                        for sequence_id in self.sequence_descriptors
                    ),
                },
        )
        loop_result_rebindings = (
            _canonicalize_non_dominating_loop_result_uses(function)
        )
        if loop_result_rebindings:
            function.metadata["loop_result_use_rebindings"] = (
                loop_result_rebindings
            )
        if self.evolution is not None and self.ssa_evolution is not None:
            self.evolution.bind_artifact(function, self.ssa_evolution)
            self.evolution.close_graph(self.ssa_evolution)
        return function, tuple(self.shortfalls)


def _canonicalize_non_dominating_loop_result_uses(
    function: Function,
) -> tuple[tuple[str, int, int, int, int], ...]:
    """Resolve stale nested LoopResult operands only when CFG proves it.

    Control construction can retain the inner spelling of a carried result in
    an expression that is finally placed after an enclosing loop.  The inner
    Phi does not dominate the enclosing loop's zero-trip exit; the enclosing
    result Phi does.  ``carried_port_values`` records the exact lexical result
    equivalence.  This pass changes an operand only when the old object fails
    dominance and the recorded replacement object satisfies it, so it is not
    a blanket value-id/SSA-version substitution.
    """

    port_values = dict(
        function.metadata.get("carried_port_values") or {}
    )
    if not port_values or not function.blocks:
        return ()

    block_names = tuple(function.blocks)
    entry = block_names[0]
    predecessors = {name: set() for name in block_names}
    for name, block in function.blocks.items():
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].add(name)

    reachable = {entry}
    pending = [entry]
    while pending:
        name = pending.pop()
        for successor in function.blocks[name].successors:
            if successor in function.blocks and successor not in reachable:
                reachable.add(successor)
                pending.append(successor)

    dominators = {
        name: ({name} if name == entry else set(reachable))
        for name in reachable
    }
    changed = True
    while changed:
        changed = False
        for name in block_names:
            if name == entry or name not in reachable:
                continue
            incoming = predecessors[name] & reachable
            parent_dominators = (
                set.intersection(*(dominators[parent] for parent in incoming))
                if incoming else set()
            )
            updated = {name} | parent_dominators
            if updated != dominators[name]:
                dominators[name] = updated
                changed = True

    argument_ids = {int(value.id) for value in function.args}
    definition_sites: dict[int, list[tuple[str, int]]] = {}
    for block_name, block in function.blocks.items():
        for instruction_index, instruction in enumerate(block.instrs):
            if instruction.res is not None:
                definition_sites.setdefault(
                    int(instruction.res.id), []
                ).append((block_name, instruction_index))

    def dominates(
        value: SSAValue,
        use_block: str,
        use_index: int,
        incoming_block: str | None,
    ) -> bool:
        value_id = int(value.id)
        if value_id in argument_ids:
            return True
        sites = definition_sites.get(value_id, ())
        # Multiple producers for one numeric identity are not valid SSA and
        # cannot be made safe by guessing which Python SSAValue object a use
        # meant.  Leave such a case untouched for structural validation.
        if len(sites) != 1:
            return False
        definition_block, definition_index = sites[0]
        target_block = incoming_block or use_block
        if target_block not in reachable:
            return False
        if definition_block != target_block:
            return definition_block in dominators[target_block]
        target_index = (
            len(function.blocks[target_block].instrs)
            if incoming_block is not None else use_index
        )
        return definition_index < target_index

    receipts = []
    for block_name, block in function.blocks.items():
        for instruction_index, instruction in enumerate(block.instrs):
            incoming_blocks = (
                tuple(instruction.attributes.get("incoming_blocks") or ())
                if str(instruction.op).casefold() == "phi" else ()
            )
            resolved_args = list(instruction.args)
            for argument_index, argument in enumerate(instruction.args):
                replacement = port_values.get(int(argument.id))
                if replacement is None or replacement is argument:
                    continue
                incoming_block = (
                    str(incoming_blocks[argument_index])
                    if argument_index < len(incoming_blocks) else None
                )
                if dominates(
                    argument, block_name, instruction_index, incoming_block
                ):
                    continue
                if not dominates(
                    replacement, block_name, instruction_index, incoming_block
                ):
                    continue
                resolved_args[argument_index] = replacement
                receipts.append((
                    str(block_name), int(instruction_index),
                    int(argument_index), int(argument.id),
                    int(replacement.id),
                ))
            instruction.args = resolved_args
    return tuple(receipts)


def lower_control_program_to_ssa(
    program: ControlProgram,
    *,
    function_name: str = "planned_control",
    first_value_id: int = 0,
    region_callees: dict[int, str] | None = None,
    region_signatures: dict[
        int, tuple[tuple[int, ...], tuple[int, ...]]
    ] | None = None,
    region_feed_meta: Mapping[int, tuple[Meta, ...]] | None = None,
    region_value_meta: Mapping[int, Meta] | None = None,
    plan_callsite_bindings: Mapping[
        int, tuple[tuple[int, ...], tuple[int, ...]]
    ] | None = None,
    value_aliases: Mapping[int, int] | None = None,
    inout_value_ids: tuple[int, ...] = (),
    output_value_ids: tuple[int, ...] = (),
    named_output_histories: Mapping[str, tuple[int, ...]] | None = None,
    value_name_histories: Mapping[str, tuple[int, ...]] | None = None,
    parameter_names: tuple[str, ...] = (),
    sequence_initializations: tuple[tuple[int, str, int], ...] = (),
    sequence_declarations: tuple[tuple[int, str, int, bool], ...] = (),
    sequence_column_dtypes: Mapping[int, tuple[str, ...]] | None = None,
    sequence_record_identities: Mapping[int, str] | None = None,
    source_sequence_ids: tuple[int, ...] = (),
    sequence_memberships: tuple[tuple[int, int, int, bool], ...] = (),
    table_lookups: tuple[tuple[int, int | tuple[int, ...], int], ...] = (),
    lexical_table_lookup_result_ids: tuple[int, ...] = (),
    table_lookup_defaults: dict[int, int | float] | None = None,
    table_stores: tuple[
        tuple[int, int | tuple[int, ...], int, int], ...
    ] = (),
    table_deletions: tuple[
        tuple[int, int | tuple[int, ...], int | None, str], ...
    ] = (),
    retained_sequence_ids: tuple[int, ...] = (),
    nested_sequence_ids: tuple[int, ...] = (),
    joined_sequence_ids: tuple[int, ...] = (),
    joined_singleton_values: Mapping[int, int] | None = None,
    nested_row_target_ids: tuple[int, ...] = (),
    selected_nested_sequence_ids: tuple[int, ...] = (),
    variant_projected_target_ids: tuple[int, ...] = (),
    region_array_feed_ids: Mapping[int, tuple[int, ...]] | None = None,
    nested_row_projections: tuple[tuple[int, int, int, str], ...] = (),
    sequence_length_values: Mapping[int, int] | None = None,
    table_region_operations: Mapping[int, tuple[tuple[str, tuple[Any, ...]], ...]] | None = None,
    table_region_post_operations: Mapping[int, tuple[tuple[str, tuple[Any, ...]], ...]] | None = None,
    table_epilogue_operations: tuple[tuple[str, tuple[Any, ...]], ...] = (),
    resolved_sequence_schemas: Mapping[int, ResolvedSequenceSchema] | None = None,
) -> tuple[Function, tuple[SSALoweringShortfall, ...]]:
    builder = _ControlSSABuilder(
        program,
        function_name=function_name,
        first_value_id=first_value_id,
        region_callees=region_callees,
        region_signatures=region_signatures,
        region_feed_meta=region_feed_meta,
        region_value_meta=region_value_meta,
        plan_callsite_bindings=plan_callsite_bindings,
        value_aliases=value_aliases,
        inout_value_ids=inout_value_ids,
        output_value_ids=output_value_ids,
        named_output_histories=named_output_histories,
        value_name_histories=value_name_histories,
        parameter_names=parameter_names,
        sequence_initializations=sequence_initializations,
        sequence_declarations=sequence_declarations,
        sequence_column_dtypes=sequence_column_dtypes,
        sequence_record_identities=sequence_record_identities,
        source_sequence_ids=source_sequence_ids,
        sequence_memberships=sequence_memberships,
        table_lookups=table_lookups,
        lexical_table_lookup_result_ids=lexical_table_lookup_result_ids,
        table_lookup_defaults=table_lookup_defaults,
        table_stores=table_stores,
        table_deletions=table_deletions,
        retained_sequence_ids=retained_sequence_ids,
        nested_sequence_ids=nested_sequence_ids,
        joined_sequence_ids=joined_sequence_ids,
        joined_singleton_values=joined_singleton_values,
        nested_row_target_ids=nested_row_target_ids,
        selected_nested_sequence_ids=selected_nested_sequence_ids,
        variant_projected_target_ids=variant_projected_target_ids,
        region_array_feed_ids=region_array_feed_ids,
        nested_row_projections=nested_row_projections,
        sequence_length_values=sequence_length_values,
        table_region_operations=table_region_operations,
        table_region_post_operations=table_region_post_operations,
        table_epilogue_operations=table_epilogue_operations,
        resolved_sequence_schemas=resolved_sequence_schemas,
    )
    builder.lower(program.root)
    for kind, arguments in builder.table_epilogue_operations:
        if kind == "delete":
            builder._emit_table_delete(*arguments)
        elif kind == "store":
            builder._emit_table_store(*arguments)
        elif kind == "pack_bits":
            builder._emit_sequence_pack_bits(*arguments)
        else:
            raise ValueError(f"unknown table epilogue operation {kind!r}")
    function, shortfalls = builder.finish()
    from .ir_sequence_tables import schedule_joined_sequence_mutations

    schedule_joined_sequence_mutations(function)
    return function, shortfalls


def _sequence_artifacts_from_control(
    function: Function,
) -> tuple[dict[str, Function], dict[str, SSASequenceTable]]:
    """Recover source-linked sequence helpers and their function-local tables."""

    helpers = {
        helper.name: helper
        for helper in function.metadata.get("sequence_helper_functions", ())
    }
    table = function.metadata.get("sequence_table")
    if not isinstance(table, SSASequenceTable) or not table.sequences:
        return helpers, {}
    tables = {function.name: table}
    for helper in helpers.values():
        metadata = helper.metadata
        sequence_ids = tuple(dict.fromkeys(
            int(sequence_id)
            for sequence_id in (
                metadata.get("sequence_id"),
                metadata.get("destination_sequence_id"),
                metadata.get("source_sequence_id"),
            )
            if sequence_id is not None
        ))
        helper_table = SSASequenceTable({
            sequence_id: replace(
                table.sequences[sequence_id], child_table_pool=None
            )
            for sequence_id in sequence_ids
            if sequence_id in table.sequences
        })
        if helper_table.sequences:
            tables[helper.name] = helper_table
    return helpers, tables


def lower_class_navigation_to_ssa(
    navigation: Any,
    *,
    namespace: str = "turing.class",
) -> IRModule:
    """Lower class navigation to ordinary, backend-supported SSA primitives.

    Class/member identities and permission names become deterministic integer
    LUT indices and permission bits. Lookup is a chain of ``Eq``/``LAnd``/
    ``Select`` operations; permission checks are ``And`` plus ``Eq``. Missing
    or denied references are ``-1``. No class-specific opcode is introduced.
    """

    mapping = (
        navigation.to_mapping()
        if hasattr(navigation, "to_mapping")
        else dict(navigation)
    )

    classes = tuple(mapping.get("classes", ()))
    permission_names = sorted({
        str(permission)
        for record in classes
        for permission in (
            *record.get("permissions", ()),
            *(
                permission
                for member in record.get("members", ())
                for permission in member.get("permissions", ())
            ),
        )
    })
    permission_bits = {
        name: 1 << index for index, name in enumerate(permission_names)
    }

    def mask(names) -> int:
        result = 0
        for name in names:
            result |= permission_bits[str(name)]
        return result

    lut_classes = []
    member_index = 0
    for class_index, record in enumerate(classes):
        members = []
        for local_index, member in enumerate(record.get("members", ())):
            members.append({
                **dict(member),
                "member_index": member_index,
                "local_slot": local_index,
                "required_mask": mask(member.get("permissions", ())),
                "kind_code": 1 if member.get("kind") == "method" else 0,
                "reference": (
                    int(member["function_reference"])
                    if member.get("function_reference") is not None
                    else local_index
                ),
            })
            member_index += 1
        lut_classes.append({
            **dict(record),
            "class_index": class_index,
            "required_mask": mask(record.get("permissions", ())),
            "members": members,
        })
    lut = {
        "classes": lut_classes,
        "permission_bits": permission_bits,
    }

    class Builder:
        def __init__(self, name: str, dtypes: tuple[str, ...]):
            self.name = name
            self.args = [SSAValue(i, dtype=dtype) for i, dtype in enumerate(dtypes)]
            self.next_id = len(self.args)
            self.instructions: list[Instr] = []

        def emit(self, operation: Handler, args=(), *, dtype="i32", **attributes):
            result = SSAValue(self.next_id, dtype=dtype)
            self.next_id += 1
            self.instructions.append(Instr(
                operation.value, list(args), result, attributes=attributes,
            ))
            return result

        def constant(self, value: int, *, dtype="i32"):
            return self.emit(Handler.Const, dtype=dtype, value=int(value))

        def finish(self, results: list[SSAValue]):
            self.instructions.append(Instr(Handler.Ret.value, results, None))
            return Function(self.name, self.args, {
                "entry": BasicBlock("entry", self.instructions),
            })

    lookup = Builder(f"{namespace}.lookup", ("i32",))
    lookup_result = lookup.constant(-1)
    for record in lut_classes:
        class_id = lookup.constant(record["class_index"])
        matches = lookup.emit(Handler.Eq, (lookup.args[0], class_id), dtype="bool")
        row = lookup.constant(record["class_index"])
        lookup_result = lookup.emit(
            Handler.Select, (matches, row, lookup_result),
        )
    lookup_function = lookup.finish([lookup_result])

    permission = Builder(
        f"{namespace}.evaluate_permission", ("i32", "i32"),
    )
    present = permission.emit(Handler.And, permission.args)
    permitted = permission.emit(
        Handler.Eq, (present, permission.args[1]), dtype="bool",
    )
    permission_function = permission.finish([permitted])

    resolve = Builder(
        f"{namespace}.resolve_member", ("i32", "i32", "i32"),
    )
    resolved_kind = resolve.constant(-1)
    resolved_reference = resolve.constant(-1)
    resolved_allowed = resolve.constant(0, dtype="bool")
    for record in lut_classes:
        class_id = resolve.constant(record["class_index"])
        class_match = resolve.emit(
            Handler.Eq, (resolve.args[0], class_id), dtype="bool",
        )
        for member in record["members"]:
            member_id = resolve.constant(member["member_index"])
            member_match = resolve.emit(
                Handler.Eq, (resolve.args[1], member_id), dtype="bool",
            )
            identity_match = resolve.emit(
                Handler.LAnd, (class_match, member_match), dtype="bool",
            )
            required = resolve.constant(
                record["required_mask"] | member["required_mask"]
            )
            present = resolve.emit(Handler.And, (resolve.args[2], required))
            permission_match = resolve.emit(
                Handler.Eq, (present, required), dtype="bool",
            )
            allowed = resolve.emit(
                Handler.LAnd, (identity_match, permission_match), dtype="bool",
            )
            kind = resolve.constant(member["kind_code"])
            reference = resolve.constant(member["reference"])
            resolved_kind = resolve.emit(
                Handler.Select, (allowed, kind, resolved_kind),
            )
            resolved_reference = resolve.emit(
                Handler.Select, (allowed, reference, resolved_reference),
            )
            resolved_allowed = resolve.emit(
                Handler.Select,
                (identity_match, permission_match, resolved_allowed),
                dtype="bool",
            )
    resolve_function = resolve.finish([
        resolved_kind, resolved_reference, resolved_allowed,
    ])

    instantiate = Builder(
        f"{namespace}.instantiate", ("i32", "i32"),
    )
    new_reference = instantiate.constant(-1)
    init_reference = instantiate.constant(-1)
    instantiate_allowed = instantiate.constant(0, dtype="bool")
    for record in lut_classes:
        class_id = instantiate.constant(record["class_index"])
        class_match = instantiate.emit(
            Handler.Eq, (instantiate.args[0], class_id), dtype="bool",
        )
        constructors = list(record.get("instantiation_functions", ()))
        constructor_members = {
            member["function_reference"]: member
            for member in record["members"]
            if member.get("function_reference") is not None
        }
        required_mask = record["required_mask"]
        for reference in constructors:
            member = constructor_members.get(reference)
            if member is not None:
                required_mask |= member["required_mask"]
        required = instantiate.constant(required_mask)
        present = instantiate.emit(Handler.And, (instantiate.args[1], required))
        permission_match = instantiate.emit(
            Handler.Eq, (present, required), dtype="bool",
        )
        allowed = instantiate.emit(
            Handler.LAnd, (class_match, permission_match), dtype="bool",
        )
        new_value = instantiate.constant(next((
            int(member["function_reference"])
            for member in record["members"]
            if member["name"] == "__new__"
            and member.get("function_reference") is not None
        ), -1))
        init_value = instantiate.constant(next((
            int(member["function_reference"])
            for member in record["members"]
            if member["name"] == "__init__"
            and member.get("function_reference") is not None
        ), -1))
        new_reference = instantiate.emit(
            Handler.Select, (allowed, new_value, new_reference),
        )
        init_reference = instantiate.emit(
            Handler.Select, (allowed, init_value, init_reference),
        )
        instantiate_allowed = instantiate.emit(
            Handler.Select,
            (class_match, permission_match, instantiate_allowed),
            dtype="bool",
        )
    instantiate_function = instantiate.finish([
        new_reference, init_reference, instantiate_allowed,
    ])

    functions = (
        lookup_function,
        instantiate_function,
        resolve_function,
        permission_function,
    )
    for function in functions:
        function.blocks["entry"].instrs[0].attributes.setdefault(
            "class_navigation_lut", lut,
        )

    # Hold the class DEFINITIONS in the module, not only the reference LUTs: each
    # class's instance-field layout and its methods (with the function-table
    # reference to each method's body). A backend reads this to emit a class's
    # methods as real, individually linkable functions -- the SSA counterpart of
    # the frontend ClassNavigationTable.
    class_definitions = []
    for record in classes:
        members = record.get("members", ())
        fields = tuple(
            SSAClassField(name=str(member["name"]), slot=int(member["slot"]))
            for member in members
            if member.get("kind") == "attribute"
            and member.get("storage") == "instance"
            and member.get("slot") is not None
        )
        methods = tuple(
            SSAClassMethod(
                name=str(member["name"]),
                function_reference=int(member["function_reference"]),
            )
            for member in members
            if member.get("function_reference") is not None
        )
        class_definitions.append(
            SSAClassDefinition(
                identity=str(record.get("identity", "")),
                fields=fields,
                methods=methods,
            )
        )
    class_table = SSAClassTable(classes=tuple(class_definitions))
    return IRModule(
        {function.name: function for function in functions},
        class_table=class_table,
    )


def find_ssa_cycles(function: Function) -> tuple[SSACycle, ...]:
    graph = nx.DiGraph()
    graph.add_nodes_from(function.blocks)
    for block in function.blocks.values():
        graph.add_edges_from(
            (block.name, successor)
            for successor in block.successors
        )
    cycles = []
    for component in nx.strongly_connected_components(graph):
        cyclic = len(component) > 1 or any(
            graph.has_edge(name, name) for name in component
        )
        if not cyclic:
            continue
        ordered = tuple(sorted(component))
        edges = tuple(sorted(
            (source, target)
            for source in component
            for target in graph.successors(source)
            if target in component
        ))
        phi_blocks = tuple(sorted(
            block_name
            for block_name in component
            if any(
                instruction.op in {
                    Handler.Phi.value,
                    Handler.Phi.value.lower(),
                }
                for instruction in function.blocks[block_name].instrs
            )
        ))
        cycles.append(
            SSACycle(function.name, ordered, edges, phi_blocks)
        )
    return tuple(sorted(cycles))


def _inject_field_slot_access(
    control_function: Function,
    *,
    self_value_id: int,
    non_self_param_ids: tuple[int, ...],
    field_ops: tuple[tuple[str, int, int], ...],
    field_count: int,
    field_const_sources: Mapping[int, Any] | None = None,
    output_value_ids: tuple[int, ...] = (),
    dtype: str = "float64",
    field_dtypes: Mapping[int, str] | None = None,
) -> tuple[Function, dict[int, tuple[int, int, str]]]:
    """Rewrite a method's control function to pass instance state as a slot arena.

    ``self`` becomes one or more sized field columns. Each logical field slot
    resolves through a deterministic ``(typed arena, local offset)`` location;
    homogeneous receivers therefore retain their one-array ABI, while mixed
    receivers use only as many typed columns as their actual fields require.
    Each field read is a ``Load`` from that location (producing the value the
    body already consumes, so it is no longer a free input); each field write
    is a ``Store`` of its source into it. The parameter list is rebuilt as
    ``(*receiver_columns, *non_self_params)``. A backend that indexes arrays
    renders each physical access directly and marks a written column
    ``intent(inout)``.

    Placement follows the schedule the graph already fixed, carried here by SSA
    data flow: a load goes right before the first instruction that consumes its
    value, a store right after the instruction that produces its source (or at
    the top, for a source that is a plain parameter). That is exactly the order
    the source wrote, so a read and a write of one slot -- ``self.x = v; return
    self.x`` -- emits store-then-load and reads back ``v`` with no special case.
    ``field_reads``/``field_writes`` arrive in schedule order so any two ops that
    land at the same point keep it.
    """

    if not control_function.blocks:
        return control_function, {}

    existing_ids = {int(value.id) for value in control_function.args}
    for block in control_function.blocks.values():
        for instruction in block.instrs:
            if instruction.res is not None:
                existing_ids.add(int(instruction.res.id))
    existing_ids.add(int(self_value_id))
    existing_ids.update(int(pid) for pid in non_self_param_ids)
    # The field ops name values that come from the graph, not the control body
    # (a read's result, a write's source), so a fresh Const/address id must
    # dodge those too or it collides with a load result.
    existing_ids.update(int(value_id) for _kind, value_id, _slot in field_ops)
    existing_ids.update(int(value_id) for value_id in output_value_ids)
    next_id = max(existing_ids, default=-1) + 1

    def fresh() -> int:
        nonlocal next_id
        value_id = next_id
        next_id += 1
        return value_id

    # A receiver is physically a set of typed columns.  The traditional one
    # arena form is the degenerate (and fastest) one-column case.  Slots keep a
    # logical index, while this table resolves that index to (arena, offset),
    # so mixed fields never need coercion or a tagged-object runtime.
    slot_dtypes = {
        int(slot): str((field_dtypes or {}).get(int(slot), dtype))
        for _kind, _value_id, slot in field_ops
    }
    column_dtypes = tuple(sorted(set(slot_dtypes.values()) or {str(dtype)}))
    slots_by_dtype = {
        column_dtype: tuple(sorted(
            slot for slot, slot_dtype in slot_dtypes.items()
            if slot_dtype == column_dtype
        ))
        for column_dtype in column_dtypes
    }
    column_arrays: dict[str, SSAValue] = {}
    for index, column_dtype in enumerate(column_dtypes):
        arena_id = int(self_value_id) if index == 0 else fresh()
        column_arrays[column_dtype] = SSAValue(
            arena_id, dtype=column_dtype,
            shape=(len(slots_by_dtype[column_dtype]),),
        )
    field_locations = {
        slot: (
            int(column_arrays[slot_dtype].id),
            slots_by_dtype[slot_dtype].index(slot),
            slot_dtype,
        )
        for slot, slot_dtype in slot_dtypes.items()
    }
    # Graph identities are not SSA versions.  A SetAttr result may therefore
    # carry the same numeric id as the receiver column which must receive it
    # (for example ``self.dt_max = computed`` can name both values ``18``).
    # Give the computed definition its own SSA identity while retaining the
    # receiver column's public/call-frame id.  The Store below then provides
    # the explicit bridge which was previously lost as an apparent self-write.
    field_write_sources: dict[int, SSAValue] = {}
    receiver_column_ids = {
        int(value.id) for value in column_arrays.values()
    }
    for kind, value_id, _slot in field_ops:
        value_id = int(value_id)
        if kind != "write" or value_id not in receiver_column_ids:
            continue
        producers = [
            instruction
            for block in control_function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
            and int(instruction.res.id) == value_id
        ]
        if len(producers) != 1:
            continue
        producer = producers[0]
        original = producer.res
        version = SSAValue(
            fresh(),
            dtype=original.dtype,
            shape=tuple(original.shape),
            device=original.device,
            accounting={
                **dict(original.accounting or {}),
                "ssa_field_write_version_of": value_id,
            },
        )
        producer.res = version
        for block in control_function.blocks.values():
            for instruction in block.instrs:
                instruction.args = [
                    version if argument is original else argument
                    for argument in instruction.args
                ]
        field_write_sources[value_id] = version
    const_sources = dict(field_const_sources or {})
    reference_sources = {
        int(value_id): dict(payload)
        for value_id, payload in const_sources.items()
        if isinstance(payload, Mapping)
        and payload.get("ssa_reference_identity") is not None
    }
    reference_slots = {
        int(slot)
        for kind, value_id, slot in field_ops
        if kind == "write" and int(value_id) in reference_sources
    }

    def slot_address(slot: int) -> tuple[list[Instr], SSAValue, str]:
        arena_id, offset, slot_dtype = field_locations[int(slot)]
        index = SSAValue(fresh(), dtype="int64")
        address = SSAValue(fresh())
        return (
            [
                Instr("Const", [], index, attributes={"value": int(offset)}),
                Instr("GetElementPtr", [column_arrays[slot_dtype], index], address),
            ],
            address,
            slot_dtype,
        )

    # The blocks are already in schedule order, so number every instruction once
    # across them; producer and first-consumer positions in that numbering are
    # where each store and load belong.
    flat = [
        (name, instruction)
        for name, block in control_function.blocks.items()
        for instruction in block.instrs
    ]
    producer_position: dict[int, int] = {}
    produced_values: dict[int, SSAValue] = {}
    first_consumer_position: dict[int, int] = {}
    for position, (_name, instruction) in enumerate(flat):
        if instruction.res is not None:
            producer_position.setdefault(int(instruction.res.id), position)
            produced_values.setdefault(int(instruction.res.id), instruction.res)
        for argument in instruction.args:
            first_consumer_position.setdefault(int(argument.id), position)

    entry_name = next(iter(control_function.blocks))

    # A method that returns a field value (a getter, ``return self.x``) has no
    # producer for that value in the control body -- it is one of these loads.
    # The control lowering could not wire it as an output because it did not yet
    # exist, so treat the return as the load's consumer and add the value to the
    # return below.
    return_ops = {Handler.Ret.value, "ret", "Return", "return"}
    return_position = next(
        (
            position
            for position in range(len(flat) - 1, -1, -1)
            if flat[position][1].op in return_ops
        ),
        None,
    )
    output_ids = tuple(int(value_id) for value_id in output_value_ids)
    output_id_set = set(output_ids)

    # (insert-position, schedule-order, home-block, instructions). The second
    # key is the field op's index in the schedule, so two ops that land at the
    # same instruction keep the order the source wrote -- a store before the
    # read that must observe it.
    insertions: list[tuple[int, int, str, list[Instr]]] = []
    field_read_ids: set[int] = {
        int(value_id) for kind, value_id, _slot in field_ops if kind == "read"
    }
    for schedule_index, (kind, value_id, slot) in enumerate(field_ops):
        prelude, address, slot_dtype = slot_address(slot)
        if kind == "read":
            value_dtype = (
                "opaque_ref" if int(slot) in reference_slots else slot_dtype
            )
            group = [
                *prelude,
                Instr(
                    "Load",
                    [address],
                    SSAValue(int(value_id), dtype=value_dtype),
                    attributes={
                        "opaque_reference_storage": True,
                        "field_slot": int(slot),
                    } if value_dtype == "opaque_ref" else {},
                ),
            ]
            position = first_consumer_position.get(int(value_id))
            if position is None and int(value_id) in output_id_set:
                position = return_position  # returned but otherwise unconsumed
            if position is None:
                continue  # a read nothing consumes has no place and no effect
        else:
            group = []
            reference = reference_sources.get(int(value_id))
            source_dtype = "opaque_ref" if reference is not None else slot_dtype
            stored_source = field_write_sources.get(
                int(value_id),
                produced_values.get(
                    int(value_id), SSAValue(int(value_id), dtype=source_dtype),
                ),
            )
            # A constant field write (self.x = None / 5 / "s") has no producer
            # in the control body, so materialise its source here -- the
            # tokenizer then turns a None/str/bytes const into a token.
            if reference is not None and int(value_id) not in producer_position:
                group.append(Instr(
                    Handler.StaticRef.value,
                    [],
                    SSAValue(int(value_id), dtype="opaque_ref"),
                    attributes={
                        "reference_handle": int(reference["reference_handle"]),
                        "reference_identity": str(
                            reference["ssa_reference_identity"]
                        ),
                        "reference_kind": str(
                            reference.get("reference_kind", "static-python")
                        ),
                        "host_resident": bool(
                            reference.get("host_resident", True)
                        ),
                    },
                ))
            elif int(value_id) in const_sources and int(value_id) not in (
                producer_position
            ):
                group.append(
                    Instr(
                        "Const",
                        [],
                        SSAValue(int(value_id), dtype=slot_dtype),
                        attributes={"value": const_sources[int(value_id)]},
                    )
                )
            group += [
                *prelude,
                Instr(
                    "Store",
                    [stored_source, address],
                    None,
                    attributes={
                        "opaque_reference_storage": True,
                        "field_slot": int(slot),
                    } if source_dtype == "opaque_ref" else {},
                ),
            ]
            producer = producer_position.get(int(stored_source.id))
            # After the producer; a parameter source has none, so at the top.
            position = producer + 1 if producer is not None else 0
        home = flat[position][0] if position < len(flat) else (
            flat[-1][0] if flat else entry_name
        )
        insertions.append((position, schedule_index, home, group))

    from collections import defaultdict

    inserts_at: dict[int, list[list[Instr]]] = defaultdict(list)
    for position, _order, _home, group in sorted(
        insertions, key=lambda item: (item[0], item[1])
    ):
        inserts_at[position].append(group)

    # Field values that are returned but were not already on the return
    # instruction (the control lowering could not see them): append them so the
    # target declares them as outputs, keeping function-output order.
    existing_return_arg_ids = (
        {int(argument.id) for argument in flat[return_position][1].args}
        if return_position is not None
        else set()
    )
    returned_field_values = [
        SSAValue(int(value_id), dtype=field_locations[
            next(slot for kind, candidate, slot in field_ops
                 if kind == "read" and int(candidate) == int(value_id))
        ][2])
        for value_id in output_ids
        if value_id in field_read_ids and value_id not in existing_return_arg_ids
    ]

    rebuilt: dict[str, list[Instr]] = {
        name: [] for name in control_function.blocks
    }
    for position, (name, instruction) in enumerate(flat):
        for group in inserts_at.get(position, ()):
            rebuilt[name].extend(group)
        if position == return_position and returned_field_values:
            instruction = Instr(
                instruction.op,
                [*instruction.args, *returned_field_values],
                instruction.res,
            )
        rebuilt[name].append(instruction)
    trailing = inserts_at.get(len(flat), ())
    if trailing:
        tail_block = flat[-1][0] if flat else entry_name
        for group in trailing:
            rebuilt[tail_block].extend(group)

    new_blocks = {
        name: BasicBlock(name, rebuilt[name])
        for name in control_function.blocks
    }

    # ``self`` first, then the non-self parameters in declared order; the read
    # field values are no longer parameters because the loads produce them.
    arguments = [column_arrays[column_dtype] for column_dtype in column_dtypes]
    seen_argument_ids = {int(argument.id) for argument in arguments}
    for argument in control_function.args:
        argument_id = int(argument.id)
        if (
            argument_id in seen_argument_ids
            or argument_id in field_read_ids
        ):
            continue
        arguments.append(argument)
        seen_argument_ids.add(argument_id)
    # A source parameter unused by the current specialized body may not have
    # been materialized by control lowering, but it remains part of the public
    # method signature. Add only those absent declared parameters; all sequence
    # and child-pool ABI arguments above retain their exact shapes and dtypes.
    for param_id in non_self_param_ids:
        if int(param_id) not in seen_argument_ids and int(param_id) not in field_read_ids:
            arguments.append(SSAValue(int(param_id), dtype=dtype))
            seen_argument_ids.add(int(param_id))

    return Function(
        control_function.name,
        arguments,
        new_blocks,
        metadata={
            **dict(control_function.metadata),
            "receiver_field_locations": tuple(sorted(
                (int(slot), *location)
                for slot, location in field_locations.items()
            )),
        },
    ), field_locations


def _schedule_loop_callsites(
    control: ControlProgram,
    hierarchy_plan: "PlanClosure | None",
    region_signatures: Mapping[int, tuple[tuple[int, ...], tuple[int, ...]]],
    region_dependency_signatures: Mapping[
        int, tuple[tuple[int, ...], tuple[int, ...]]
    ] | None = None,
) -> tuple[ControlProgram, dict[int, tuple[tuple[int, ...], tuple[int, ...]]]]:
    """Install this closure's PlanCalls as ordinary scheduled statements.

    The hierarchy plan already interleaves calls and numerical regions. Carry
    that order into local Control IR: a call precedes the next retained region,
    or remains at its enclosing loop/closure boundary when it is trailing.
    The dependency pass below can then order calls by their exact argument and
    result identities just as it orders numerical regions. Linking supplies
    the final physical frame later; it does not rediscover execution position.
    """

    if hierarchy_plan is None:
        return control, {}

    bindings: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    planned_calls: dict[int, PlanCall] = {
        int(item.callsite_id): item
        for item in hierarchy_plan.items
        if isinstance(item, PlanCall)
    }
    for callsite_id, item in planned_calls.items():
        argument_ids = tuple(
            int(caller_id)
            for caller_id, _callee_id in item.argument_bindings
        ) or tuple(int(v) for v in item.argument_value_ids)
        result_ids = tuple(
            int(caller_id)
            for _callee_id, caller_id in item.result_bindings
        ) or tuple(int(v) for v in item.result_value_ids)
        bindings[callsite_id] = (argument_ids, result_ids)

    # Value dependencies are not the whole execution-order contract.  A
    # mutable aggregate write can publish a fresh graph identity while the
    # following call is deliberately rebound to the same resident storage.
    # At that boundary the call arguments and the write-region outputs need
    # not share an SSA value id, but their order is still explicit in the
    # hierarchy.  Preserve that order as a precedence chain among the
    # hierarchy statements which survive in this control closure.  Structural
    # producers not represented by a PlanCall/region remain free to move ahead
    # when the ordinary dependency analysis below proves that necessary.
    hierarchy_statement_rank: dict[tuple[str, int], int] = {}
    for position, item in enumerate(hierarchy_plan.items):
        if isinstance(item, PlanCall):
            hierarchy_statement_rank[("call", int(item.callsite_id))] = position
        elif (
            isinstance(item, PlanClosure)
            and item.name.startswith("region_")
        ):
            hierarchy_statement_rank[(
                "region", int(item.name.split("_", 1)[1])
            )] = position

    def scheduled_region(block: Any) -> int | None:
        if not isinstance(block, StatementBlock) or len(block.lines) != 1:
            return None
        match = _REGION_MARKER.fullmatch(str(block.lines[0]))
        return None if match is None else int(match.group(1))

    retained_regions: set[int] = set()
    present_loop_ids: set[int] = set()

    def discover_control(block: Any) -> None:
        region = scheduled_region(block)
        if region is not None:
            retained_regions.add(region)
        if isinstance(block, SequenceBlock):
            for child in block.blocks:
                discover_control(child)
        elif isinstance(block, ConditionalBlock):
            discover_control(block.body)
            if block.orelse is not None:
                discover_control(block.orelse)
        elif isinstance(block, LoopBlock):
            induction = str(block.induction)
            if induction.startswith("iteration_"):
                try:
                    present_loop_ids.add(int(induction[len("iteration_"):]))
                except ValueError:
                    pass
            discover_control(block.body)
        elif isinstance(block, WhileBlock):
            if block.source_loop_node_id is not None:
                present_loop_ids.add(int(block.source_loop_node_id))
            discover_control(block.condition)
            discover_control(block.body)
        elif isinstance(block, CallBlock):
            discover_control(block.callee)

    discover_control(control.root)
    calls_before: dict[int, list[int]] = {}
    replaced_projection_regions: set[int] = set()
    pending: list[int] = []

    def is_call_projection_region(
        closure: "PlanClosure", call: PlanCall,
    ) -> bool:
        """Whether ``closure`` only unpacks outputs already bound by ``call``."""

        result_ids = set(bindings[int(call.callsite_id)][1])
        if not result_ids:
            return False
        projected_ids: set[int] = set()
        for line in closure.items:
            opcode = str(getattr(line, "opcode", "")).casefold()
            if opcode == "const":
                continue
            inputs = tuple(map(int, getattr(line, "inputs", ())))
            outputs = tuple(map(int, getattr(line, "outputs", ())))
            if (
                opcode != "indexed"
                or not inputs
                or inputs[0] != int(call.callsite_id)
                or not outputs
                or not set(outputs).issubset(result_ids)
            ):
                return False
            projected_ids.update(outputs)
        return projected_ids == result_ids

    for item in hierarchy_plan.items:
        if isinstance(item, PlanCall):
            pending.append(int(item.callsite_id))
            continue
        if not (
            isinstance(item, PlanClosure)
            and item.name.startswith("region_")
        ):
            continue
        region = int(item.name.split("_", 1)[1])
        if region in retained_regions and pending:
            calls_before.setdefault(region, []).extend(pending)
            if is_call_projection_region(item, planned_calls[pending[-1]]):
                replaced_projection_regions.add(region)
            pending = []
    trailing_at_loop: dict[int, list[int]] = {}
    trailing_at_root: list[int] = []
    for callsite_id in pending:
        item = planned_calls[callsite_id]
        loop_id = next((
            int(candidate)
            for candidate in reversed(item.enclosing_loop_ids)
            if int(candidate) in present_loop_ids
        ), None)
        if loop_id is None:
            trailing_at_root.append(callsite_id)
        else:
            trailing_at_loop.setdefault(loop_id, []).append(callsite_id)

    def marker(callsite_ids: Iterable[int]) -> StatementBlock:
        return StatementBlock(tuple(
            f"__plan_callsite_{int(callsite_id)}__"
            for callsite_id in callsite_ids
        ))

    def sequence(*blocks: Any) -> SequenceBlock:
        return SequenceBlock(tuple(
            child
            for block in blocks
            for child in (
                block.blocks if isinstance(block, SequenceBlock) else (block,)
            )
        ))

    def rebuild(block: Any) -> Any:
        region = scheduled_region(block)
        if region is not None:
            before = calls_before.get(region, ())
            if region in replaced_projection_regions:
                return marker(before)
            return block if not before else sequence(marker(before), block)
        if isinstance(block, SequenceBlock):
            return sequence(*(rebuild(child) for child in block.blocks))
        if isinstance(block, LoopBlock):
            rebuilt_body = rebuild(block.body)
            induction = str(block.induction)
            loop_id = (
                int(induction[len("iteration_"):])
                if induction.startswith("iteration_")
                and induction[len("iteration_"):].isdigit()
                else None
            )
            trailing = trailing_at_loop.get(loop_id, ())
            if trailing:
                rebuilt_body = sequence(rebuilt_body, marker(trailing))
            return replace(block, body=rebuilt_body)
        if isinstance(block, WhileBlock):
            rebuilt_condition = rebuild(block.condition)
            rebuilt_body = rebuild(block.body)
            trailing = trailing_at_loop.get(
                None if block.source_loop_node_id is None
                else int(block.source_loop_node_id),
                (),
            )
            if trailing:
                rebuilt_body = sequence(rebuilt_body, marker(trailing))
            return replace(
                block, condition=rebuilt_condition, body=rebuilt_body
            )
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=rebuild(block.body),
                orelse=(
                    rebuild(block.orelse)
                    if block.orelse is not None else None
                ),
            )
        return block

    rebuilt_root = rebuild(control.root)
    if trailing_at_root:
        rebuilt_root = sequence(rebuilt_root, marker(trailing_at_root))
    rebuilt_control = replace(control, root=rebuilt_root)

    def produced_sequences(block: Any) -> set[int]:
        """Resident sequence arenas populated by one control subtree."""

        produced: set[int] = set()
        if isinstance(block, SequenceMutationBlock):
            produced.add(int(block.mutation.sequence_value_id))
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                produced.update(produced_sequences(child))
        elif isinstance(block, ConditionalBlock):
            produced.update(produced_sequences(block.body))
            if block.orelse is not None:
                produced.update(produced_sequences(block.orelse))
        elif isinstance(block, (LoopBlock, WhileBlock)):
            produced.update(
                int(mutation.sequence_value_id)
                for mutation in block.sequence_mutations
            )
            if isinstance(block, WhileBlock):
                produced.update(produced_sequences(block.condition))
            produced.update(produced_sequences(block.body))
        elif isinstance(block, CallBlock):
            produced.update(produced_sequences(block.callee))
        return produced


    def dependency_signature(
        block: Any,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Return exact straight-line inputs and publications for a block."""

        if isinstance(block, StatementBlock):
            region_index = scheduled_region(block)
            if region_index is not None:
                return (region_dependency_signatures or {}).get(
                    region_index,
                    region_signatures.get(region_index, ((), ())),
                )
            callsites = []
            for line in block.lines:
                match = _CALLSITE_MARKER.fullmatch(str(line))
                if match is None:
                    return None
                callsites.append(int(match.group(1)))
            if not callsites:
                return None
            return (
                tuple(dict.fromkeys(
                    value_id
                    for callsite_id in callsites
                    for value_id in bindings.get(callsite_id, ((), ()))[0]
                )),
                tuple(dict.fromkeys(
                    value_id
                    for callsite_id in callsites
                    for value_id in bindings.get(callsite_id, ((), ()))[1]
                )),
            )
        if isinstance(block, SequenceMutationBlock):
            mutation = block.mutation
            return (
                tuple(map(int, mutation.argument_value_ids)),
                (
                    int(mutation.effect_node_id),
                    int(mutation.sequence_value_id),
                ),
            )
        if isinstance(block, SequenceQueryBlock):
            return (
                tuple(dict.fromkeys((
                    int(block.sequence_value_id),
                    *(
                        () if block.default_value_id is None
                        else (int(block.default_value_id),)
                    ),
                ))),
                tuple(dict.fromkeys((
                    int(block.result_value_id),
                    *map(int, block.result_alias_ids),
                ))),
            )
        if isinstance(block, (LoopBlock, WhileBlock)):
            # A loop is one scheduling unit, but it is not dependency-free.
            # Pure regions immediately outside it can define loop-invariant
            # values consumed anywhere in its body (row-neighbour indices in
            # the fluid stencil are the minimal example).  Hiding those feeds
            # lets the stable outer schedule leave their producer after the
            # loop, creating uses which no definition dominates.
            consumed: set[int] = set()
            produced: set[int] = set(produced_sequences(block))

            def collect_surface(candidate: Any) -> None:
                if isinstance(candidate, StatementBlock):
                    region_index = scheduled_region(candidate)
                    signature = (
                        None if region_index is None
                        else (region_dependency_signatures or {}).get(
                            region_index,
                            region_signatures.get(region_index, ((), ())),
                        )
                    )
                    if signature is not None:
                        consumed.update(map(int, signature[0]))
                        produced.update(map(int, signature[1]))
                    return
                if isinstance(candidate, SequenceBlock):
                    for child in candidate.blocks:
                        collect_surface(child)
                    return
                if isinstance(candidate, ConditionalBlock):
                    if candidate.predicate_value_id is not None:
                        consumed.add(int(candidate.predicate_value_id))
                    collect_surface(candidate.body)
                    if candidate.orelse is not None:
                        collect_surface(candidate.orelse)
                    return
                if isinstance(candidate, (LoopBlock, WhileBlock)):
                    if isinstance(candidate, WhileBlock):
                        consumed.add(int(candidate.predicate_value_id))
                        collect_surface(candidate.condition)
                    collect_surface(candidate.body)
                    return
                signature = dependency_signature(candidate)
                if signature is not None:
                    consumed.update(map(int, signature[0]))
                    produced.update(map(int, signature[1]))

            if isinstance(block, WhileBlock):
                consumed.add(int(block.predicate_value_id))
                collect_surface(block.condition)
            collect_surface(block.body)
            consumed.update(
                int(initial_id)
                for _updated_id, initial_id in block.carried_aliases
            )
            produced.update(
                int(port_id) for port_id, _initial, _updated
                in block.result_ports
            )
            return (
                tuple(sorted(consumed - produced)),
                tuple(sorted(produced)),
            )
        return None

    def hierarchy_statement_position(block: Any) -> int | None:
        """Return this scheduled statement's position in the hierarchy."""

        if not isinstance(block, StatementBlock):
            return None
        region_index = scheduled_region(block)
        if region_index is not None:
            return hierarchy_statement_rank.get(("region", region_index))
        callsites = []
        for line in block.lines:
            match = _CALLSITE_MARKER.fullmatch(str(line))
            if match is None:
                return None
            callsites.append(int(match.group(1)))
        positions = tuple(
            hierarchy_statement_rank[("call", callsite_id)]
            for callsite_id in callsites
            if ("call", callsite_id) in hierarchy_statement_rank
        )
        return min(positions) if positions else None

    def dependency_order(block: Any) -> Any:
        """Stable-toposort exact structural dependencies in one lexical arm.

        Unrecognised control is a hard barrier.  Recognised effects may move
        only when an explicit value dependency proves the lexical order
        impossible; otherwise the stable queue retains authored order.
        """

        if isinstance(block, SequenceBlock):
            children = [dependency_order(child) for child in block.blocks]
            ordered: list[Any] = []
            run: list[Any] = []

            def flush() -> None:
                if len(run) < 2:
                    ordered.extend(run)
                    run.clear()
                    return
                signatures = [dependency_signature(child) for child in run]
                producers: dict[int, list[int]] = {}
                for position, signature in enumerate(signatures):
                    assert signature is not None
                    for output in signature[1]:
                        producers.setdefault(int(output), []).append(position)
                dependencies: list[set[int]] = []
                for position, signature in enumerate(signatures):
                    assert signature is not None
                    required: set[int] = set()
                    for feed in signature[0]:
                        candidates = producers.get(int(feed), ())
                        prior = [candidate for candidate in candidates if candidate < position]
                        later = [candidate for candidate in candidates if candidate > position]
                        if prior:
                            required.add(max(prior))
                        elif later:
                            required.add(min(later))
                    dependencies.append(required)
                hierarchy_positions = [
                    hierarchy_statement_position(child) for child in run
                ]
                ranked = sorted(
                    (
                        (int(hierarchy_position), position)
                        for position, hierarchy_position
                        in enumerate(hierarchy_positions)
                        if hierarchy_position is not None
                    ),
                )
                for (_prior_rank, prior), (_rank, current) in zip(
                    ranked, ranked[1:]
                ):
                    dependencies[current].add(prior)
                remaining = list(range(len(run)))
                placed: set[int] = set()
                while remaining:
                    ready = next((
                        position for position in remaining
                        if dependencies[position].issubset(placed)
                    ), None)
                    if ready is None:
                        # Conflicting mutation versions cannot be repaired by
                        # guessing. Preserve the planner order for the cycle.
                        ordered.extend(run[position] for position in remaining)
                        break
                    ordered.append(run[ready])
                    placed.add(ready)
                    remaining.remove(ready)
                run.clear()

            for child in children:
                if dependency_signature(child) is None:
                    flush()
                    ordered.append(child)
                else:
                    run.append(child)
            flush()
            return replace(block, blocks=tuple(ordered))
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=dependency_order(block.body),
                orelse=(
                    None if block.orelse is None
                    else dependency_order(block.orelse)
                ),
            )
        if isinstance(block, LoopBlock):
            return replace(block, body=dependency_order(block.body))
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=dependency_order(block.condition),
                body=dependency_order(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=dependency_order(block.callee))
        return block

    return replace(
        rebuilt_control,
        root=dependency_order(rebuilt_control.root),
    ), bindings


def _materialize_control_constants(
    function: Function,
    constant_values: Mapping[int, Any],
    *,
    value_dtypes: Mapping[int, str],
) -> Function:
    """Materialize authored graph literals instead of exposing ABI inputs.

    Numerical regions legitimately receive literals as call arguments.  The
    enclosing control function owns those literals and must produce them
    before making the region call.  Keeping them in its signature would make
    every source constant an invented caller-supplied parameter.
    """

    if not function.blocks or not constant_values:
        return function
    argument_ids = {int(argument.id) for argument in function.args}
    existing: dict[int, Instr] = {}
    for block in function.blocks.values():
        retained = []
        for instruction in block.instrs:
            value_id = (
                None if instruction.res is None
                else int(instruction.res.id)
            )
            if (
                value_id in constant_values
                and str(instruction.op).casefold() in {
                    "const", "constant", "nonevalue",
                }
            ):
                literal = constant_values[value_id]
                matches = (
                    str(instruction.op).casefold() == "nonevalue"
                    and literal is None
                    and not instruction.attributes
                ) or (
                    str(instruction.op).casefold() in {"const", "constant"}
                    and instruction.attributes.get("value") == literal
                )
                if not matches:
                    raise ValueError(
                        "canonical constant identity has conflicting literals: "
                        f"value={value_id}"
                    )
                if literal is None and str(instruction.op).casefold() != "nonevalue":
                    instruction.op = "NoneValue"
                    instruction.attributes = {}
                    if instruction.res is not None:
                        instruction.res.dtype = "none"
                existing.setdefault(value_id, instruction)
                continue
            retained.append(instruction)
        block.instrs[:] = retained
    materialized_ids = tuple(sorted({
        int(value_id)
        for value_id in constant_values
        if int(value_id) in argument_ids or int(value_id) in existing
    }))
    if not materialized_ids:
        return function
    entry = next(iter(function.blocks.values()))
    materializations = []
    for value_id in materialized_ids:
        resident = existing.get(value_id)
        if resident is not None:
            materializations.append(resident)
            continue
        literal = constant_values[value_id]
        is_none = literal is None
        materializations.append(Instr(
            "NoneValue" if is_none else "Const", [],
            SSAValue(
                value_id,
                dtype=(
                    "none" if is_none
                    else str(value_dtypes.get(value_id) or "float64")
                ),
                accounting={"authored_constant": True},
            ),
            attributes={} if is_none else {
                "value": copy.deepcopy(literal)
            },
        ))
    entry.instrs[0:0] = materializations
    materialized = set(materialized_ids)
    function.args = [
        argument for argument in function.args
        if int(argument.id) not in materialized
    ]
    return function


def _install_loop_owned_table_queries(
    control: ControlProgram,
    table_lookups: Iterable[tuple[int, int | tuple[int, ...], int]],
    loop_owners: Mapping[int, int],
    *,
    excluded_result_ids: Iterable[int] = (),
    globally_mutated_sequence_ids: Iterable[int] = (),
) -> tuple[ControlProgram, tuple[int, ...], tuple[tuple[int, str], ...]]:
    """Place keyed reads only under their exact retained source loop.

    The graph identifies both the lookup's key value and the comprehension
    node which defines that key.  A retained LoopBlock carries that same
    source node id.  This is sufficient to place a pure read after the loop's
    projected target load, provided the queried sequence is not mutated in the
    loop.  Missing/ambiguous owners are left untouched and reported; numeric
    proximity is never used as a scheduling surrogate.
    """

    excluded = set(map(int, excluded_result_ids))
    globally_mutated = set(map(int, globally_mutated_sequence_ids))
    by_loop: dict[int, list[tuple[int, tuple[int, ...], int]]] = {}
    for result_id, query_ids, sequence_id in table_lookups:
        result_id = int(result_id)
        if result_id in excluded:
            continue
        keys = tuple(map(
            int, query_ids if isinstance(query_ids, tuple) else (query_ids,),
        ))
        owners = {
            int(loop_owners[key]) for key in keys if key in loop_owners
        }
        if len(owners) != 1 or any(key not in loop_owners for key in keys):
            continue
        owner, = owners
        by_loop.setdefault(owner, []).append((
            result_id, keys, int(sequence_id),
        ))

    installed: list[int] = []
    refusals: list[tuple[int, str]] = []

    def mutated_sequences(block: ControlBlock) -> set[int]:
        found: set[int] = set()
        if isinstance(block, SequenceMutationBlock):
            found.add(int(block.mutation.sequence_value_id))
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                found.update(mutated_sequences(child))
        elif isinstance(block, ConditionalBlock):
            found.update(mutated_sequences(block.body))
            if block.orelse is not None:
                found.update(mutated_sequences(block.orelse))
        elif isinstance(block, LoopBlock):
            found.update(
                int(item.sequence_value_id) for item in block.sequence_mutations
            )
            found.update(mutated_sequences(block.body))
        elif isinstance(block, WhileBlock):
            found.update(
                int(item.sequence_value_id) for item in block.sequence_mutations
            )
            found.update(mutated_sequences(block.condition))
            found.update(mutated_sequences(block.body))
        elif isinstance(block, CallBlock):
            found.update(mutated_sequences(block.callee))
        elif isinstance(block, ParallelDeployment):
            for lane in block.lanes:
                found.update(mutated_sequences(lane))
        elif isinstance(block, StateMachineTick):
            for _case, body in block.cases:
                found.update(mutated_sequences(body))
            if block.default is not None:
                found.update(mutated_sequences(block.default))
        return found

    def rewrite(block: ControlBlock) -> ControlBlock:
        if isinstance(block, SequenceBlock):
            return replace(block, blocks=tuple(map(rewrite, block.blocks)))
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=rewrite(block.body),
                orelse=(
                    None if block.orelse is None else rewrite(block.orelse)
                ),
            )
        if isinstance(block, LoopBlock):
            body = rewrite(block.body)
            owned = tuple(by_loop.get(int(
                block.source_loop_node_id
                if block.source_loop_node_id is not None else -1
            ), ()))
            if not owned:
                return replace(block, body=body)
            mutated = mutated_sequences(body) | {
                int(item.sequence_value_id) for item in block.sequence_mutations
            } | globally_mutated
            queries = []
            for result_id, keys, sequence_id in owned:
                if sequence_id in mutated:
                    refusals.append((
                        result_id,
                        f"sequence {sequence_id} is mutable in its owner loop",
                    ))
                    continue
                queries.append(SequenceQueryBlock(
                    result_value_id=result_id,
                    sequence_value_id=sequence_id,
                    operation="lookup",
                    source_call_node_id=result_id,
                    producer_loop_node_id=block.source_loop_node_id,
                    key_value_ids=keys,
                ))
                installed.append(result_id)
            if not queries:
                return replace(block, body=body)
            body_blocks = (
                body.blocks if isinstance(body, SequenceBlock) else (body,)
            )
            return replace(
                block,
                body=SequenceBlock((*queries, *body_blocks)),
            )
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=rewrite(block.condition),
                body=rewrite(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=rewrite(block.callee))
        if isinstance(block, ParallelDeployment):
            return replace(block, lanes=tuple(map(rewrite, block.lanes)))
        if isinstance(block, StateMachineTick):
            return replace(
                block,
                cases=tuple(
                    (case, rewrite(body)) for case, body in block.cases
                ),
                default=(
                    None if block.default is None else rewrite(block.default)
                ),
            )
        return block

    return (
        replace(control, root=rewrite(control.root)),
        tuple(dict.fromkeys(installed)),
        tuple(refusals),
    )


def lower_control_sections_to_ssa(
    control: ControlProgram,
    *,
    hierarchy_plan: PlanClosure | None = None,
    preloaded_value_aliases: Mapping[int, int] | None = None,
    control_name: str = "planned_control",
    identity_table: Mapping[str, tuple[int, ...]] | None = None,
    function_outputs: tuple[str, ...] = (),
    function_parameters: tuple[str, ...] = (),
    value_dtypes: Mapping[int, str] | None = None,
    value_shapes: Mapping[int, tuple[int, ...]] | None = None,
    constant_values: Mapping[int, Any] | None = None,
    required_output_value_ids: tuple[int, ...] = (),
    region_output_value_ids: Mapping[int, tuple[int, ...]] | None = None,
    record_field_write_value_ids: tuple[int, ...] = (),
    self_value_id: int | None = None,
    field_ops: tuple[tuple[str, int, int], ...] = (),
    field_const_sources: Mapping[int, Any] | None = None,
    field_count: int = 0,
    field_names: tuple[str, ...] = (),
    record_identity: str | None = None,
    record_field_dtypes: Mapping[str, str] | None = None,
    record_field_mutability: Mapping[str, bool] | None = None,
    sequence_initializations: tuple[tuple[int, str, int], ...] = (),
    field_aliases: tuple[tuple[int, int], ...] = (),
    sequence_declarations: tuple[tuple[int, str, int, bool], ...] = (),
    sequence_column_dtypes: Mapping[int, tuple[str, ...]] | None = None,
    sequence_record_identities: Mapping[int, str] | None = None,
    source_sequence_ids: tuple[int, ...] = (),
    sequence_memberships: tuple[tuple[int, int, int, bool], ...] = (),
    table_lookups: tuple[tuple[int, int | tuple[int, ...], int], ...] = (),
    table_lookup_loop_owners: Mapping[int, int] | None = None,
    table_lookup_defaults: dict[int, int | float] | None = None,
    table_stores: tuple[
        tuple[int, int | tuple[int, ...], int, int], ...
    ] = (),
    table_deletions: tuple[
        tuple[int, int | tuple[int, ...], int | None, str], ...
    ] = (),
    retained_sequence_ids: tuple[int, ...] = (),
    nested_sequence_ids: tuple[int, ...] = (),
    joined_sequence_ids: tuple[int, ...] = (),
    joined_singleton_values: Mapping[int, int] | None = None,
    nested_record_fields: tuple[tuple[int, str, int], ...] = (),
    sequence_augassigns: tuple[tuple[int, int, int], ...] = (),
    sequence_concats: tuple[
        tuple[int, int, int, str, int | None, int | None], ...
    ] = (),
    sequence_append_fills: tuple[
        tuple[int, int, int | float | bool | None, int, int], ...
    ] = (),
    sequence_append_slices: tuple[
        tuple[int, int, int, int, int, int], ...
    ] = (),
    sequence_bit_packs: tuple[
        tuple[int, int, int, tuple[int, ...]], ...
    ] = (),
    sequence_prepends: tuple[tuple[int, int, int, int, int], ...] = (),
    sequence_prepend_packed_calls: tuple[
        tuple[int, int, int, int, int, int, int], ...
    ] = (),
    sequence_inplace_bit_pack_calls: tuple[
        tuple[int, int, int, int, int], ...
    ] = (),
    sequence_row_operations: tuple[tuple[Any, ...], ...] = (),
    nested_row_projections: tuple[tuple[int, int, int, str], ...] = (),
    sequence_length_values: Mapping[int, int] | None = None,
    string_table: Any = None,
    tensor_ssa_reference: Any = None,
    resolved_sequence_schemas: Mapping[int, ResolvedSequenceSchema] | None = None,
) -> tuple[
    IRModule,
    tuple[SSALoweringShortfall, ...],
    dict[str, tuple[SSAValue, ...]],
]:
    """Lower one method's control + planner regions to SSA -- no numeric projection.

    This is the whole-object emission path.  A class method is control (field
    get/set, calls, returns) plus zero or more flat operator regions the planner
    already carved out.  Neither needs a fused numerical program: the regions
    come straight from the hierarchy plan via ``plan_region_to_ssa_instrs`` and
    the control program lowers directly.  Nothing here builds or validates a
    ``FusedProgram``, so a method with no numeric region (a void ``__init__``)
    and a method with one (``scale``'s ``mul``) both lower the same way.

    Returns the module, any shortfalls, and ``section_outputs`` -- the output
    SSA values of each region function, which the target must declare as that
    function's ``intent(out)`` dummies (a region has no explicit return).
    """

    functions: dict[str, Function] = {}
    region_callees: dict[int, str] = {}
    region_signatures: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    region_feed_meta: dict[int, tuple[Meta, ...]] = {}
    region_value_meta: dict[int, Meta] = {}
    nested_row_target_ids: set[int] = set()
    selected_nested_sequence_ids: set[int] = set()
    variant_projected_target_ids: set[int] = set()
    region_array_feed_ids: dict[int, set[int]] = {}
    section_outputs: dict[str, tuple[SSAValue, ...]] = {}
    plan_value_watermark = 0

    def _watermark(closure: "PlanClosure") -> None:
        nonlocal plan_value_watermark
        for _vid, _shape, _dtype in closure.value_shapes:
            plan_value_watermark = max(plan_value_watermark, int(_vid) + 1)
        for item in closure.items:
            if isinstance(item, PlanLine):
                for value_id in (*item.inputs, *item.outputs):
                    plan_value_watermark = max(
                        plan_value_watermark, int(value_id) + 1
                    )
            elif isinstance(item, PlanClosure):
                _watermark(item)
            elif isinstance(item, PlanCall):
                for value_id in (
                    *item.argument_value_ids, *item.result_value_ids,
                    *(caller for caller, _callee in item.argument_bindings),
                    *(caller for _callee, caller in item.result_bindings),
                ):
                    plan_value_watermark = max(
                        plan_value_watermark, int(value_id) + 1
                    )

    if hierarchy_plan is not None:
        _watermark(hierarchy_plan)

    shortfalls: list[SSALoweringShortfall] = []
    table_sequence_ids = {
        int(sequence_id)
        for sequence_id, policy, column_count, _writable
        in sequence_declarations
        if str(policy) == "unique" and int(column_count) > 1
    }
    declared_sequence_ids = {
        int(sequence_id)
        for sequence_id, _policy, _column_count, _writable
        in sequence_declarations
    }
    table_lookup_result_ids = {
        int(result_id) for result_id, _query_id, _sequence_id in table_lookups
    }
    table_store_effect_ids = {
        int(effect_id)
        for effect_id, _key_id, _value_id, _sequence_id in table_stores
    }
    sequence_fills = {
        int(sequence_id): (
            int(sequence_id),
            ast.literal_eval(str(policy).split(";", 1)[0].split("=", 1)[1]),
            int(str(policy).split("count=", 1)[1]),
        )
        for sequence_id, policy, _column_count in sequence_initializations
        if str(policy).startswith("fill=") and ";count=" in str(policy)
    }
    sequence_augment_by_result = {
        int(result_id): (int(destination_id), int(source_id))
        for result_id, destination_id, source_id in sequence_augassigns
    }
    sequence_concat_by_result = {
        int(result_id): (
            int(lhs_id), int(rhs_id), str(kind),
            None if lhs_scalar is None else int(lhs_scalar),
            None if rhs_scalar is None else int(rhs_scalar),
        )
        for (
            result_id, lhs_id, rhs_id, kind, lhs_scalar, rhs_scalar,
        ) in sequence_concats
    }
    sequence_append_fill_by_result = {
        int(result_id): (
            int(destination_id), literal, int(count_id)
        )
        for result_id, destination_id, literal, count_id, _source_result_id
        in sequence_append_fills
    }
    append_fill_source_result_ids = {
        int(source_result_id)
        for _result_id, _destination_id, _literal, _count_id, source_result_id
        in sequence_append_fills
    }
    sequence_append_slice_by_result = {
        int(result_id): (
            int(destination_id),
            int(source_id),
            None if lower_id is None else int(lower_id),
            None if upper_id is None else int(upper_id),
        )
        for (
            result_id, destination_id, source_id, lower_id, upper_id,
            _source_result_id,
        ) in sequence_append_slices
    }
    append_slice_source_result_ids = {
        int(source_result_id)
        for (
            _result_id, _destination_id, _source_id, _lower_id, _upper_id,
            source_result_id,
        ) in sequence_append_slices
        if source_result_id is not None
    }
    for (
        _result_id, destination_id, source_id, _lower_id, _upper_id, _slice_id,
    ) in sequence_append_slices:
        for value_id in (int(destination_id), int(source_id)):
            region_value_meta[value_id] = Meta((), "int")
            for history in (identity_table or {}).values():
                if value_id in tuple(map(int, history)):
                    for alias_id in history:
                        region_value_meta[int(alias_id)] = Meta((), "int")
    for (
        _result_id, _destination_id, _source_id, lower_id, upper_id, _slice_id,
    ) in sequence_append_slices:
        # Slice bounds are address arithmetic.  State that contract before the
        # numerical region producing e.g. ``i + width`` is typed, so its result
        # and operands remain integer across the region/helper boundary.
        # Whole-sequence extends carry no authored bounds (None).
        if lower_id is not None:
            region_value_meta[int(lower_id)] = Meta((), "int")
        if upper_id is not None:
            region_value_meta[int(upper_id)] = Meta((), "int")
    bit_pack_consumed_ids = {
        int(value_id)
        for _destination_id, _source_id, _width_id, consumed_ids
        in sequence_bit_packs
        for value_id in consumed_ids
    }
    for destination_id, source_id, width_id, _consumed_ids in sequence_bit_packs:
        region_value_meta[int(destination_id)] = Meta((), "int")
        region_value_meta[int(source_id)] = Meta((), "int")
        region_value_meta[int(width_id)] = Meta((), "int")
    prepend_by_result = {
        int(concat_result_id): (int(sequence_id), int(value_id))
        for (
            _store_result_id, sequence_id, value_id, concat_result_id,
            _tail_result_id,
        ) in sequence_prepends
    }
    prepend_store_result_ids = {
        int(store_result_id)
        for store_result_id, _sequence_id, _value_id, _concat_id, _tail_id
        in sequence_prepends
    }
    packed_concat_ids = {
        int(concat_result_id)
        for (
            _store_result_id, _destination_id, _prefix_id, _source_id,
            concat_result_id, _tail_result_id, _callsite_id,
        ) in sequence_prepend_packed_calls
    }
    packed_store_ids = {
        int(store_result_id)
        for (
            store_result_id, _destination_id, _prefix_id, _source_id,
            _concat_result_id, _tail_result_id, _callsite_id,
        ) in sequence_prepend_packed_calls
    }
    packed_by_concat = {
        int(concat_result_id): (
            int(destination_id), int(source_id), int(prefix_id),
            int(callsite_id),
        )
        for (
            _store_result_id, destination_id, prefix_id, source_id,
            concat_result_id, _tail_result_id, callsite_id,
        ) in sequence_prepend_packed_calls
    }
    packed_by_store = {
        int(store_result_id): (
            int(destination_id), int(source_id), int(prefix_id),
            int(callsite_id),
        )
        for (
            store_result_id, destination_id, prefix_id, source_id,
            _concat_result_id, _tail_result_id, callsite_id,
        ) in sequence_prepend_packed_calls
    }
    # A packed-call replacement supersedes the scalar-only prepend view of the
    # same authored concatenation/store pair.
    for concat_id in packed_concat_ids:
        prepend_by_result.pop(concat_id, None)
    prepend_store_result_ids.difference_update(packed_store_ids)
    inplace_pack_by_result = {
        int(result_id): (
            int(sequence_id), int(width), int(callee_reference),
            int(callsite_id),
        )
        for result_id, sequence_id, width, callee_reference, callsite_id
        in sequence_inplace_bit_pack_calls
    }
    for _result_id, sequence_id, _width, _reference, _callsite_id in (
        sequence_inplace_bit_pack_calls
    ):
        region_value_meta[int(sequence_id)] = Meta((), "int")
    for _result_id, destination_id, literal, _count_id, _source_id in (
        sequence_append_fills
    ):
        element_dtype = "int" if isinstance(literal, int) else "float64"
        destination_id = int(destination_id)
        region_value_meta[destination_id] = Meta((), element_dtype)
        for history in (identity_table or {}).values():
            if destination_id not in tuple(map(int, history)):
                continue
            for value_id in history:
                region_value_meta[int(value_id)] = Meta((), element_dtype)
    # Sequence replication is a structural memory contract.  Its destination
    # element and requested extent must agree in every numerical region and in
    # the control/helper function that surrounds those regions.
    for sequence_id, literal, count_id in sequence_fills.values():
        element_dtype = (
            "bool" if isinstance(literal, bool)
            else "int" if isinstance(literal, int)
            else "float64" if isinstance(literal, float)
            else "int64" if literal is None
            else "unknown"
        )
        region_value_meta[int(sequence_id)] = Meta((), element_dtype)
        region_value_meta[int(count_id)] = Meta((), "int")
    for _iterable_id, target_id, _induction_name, projection in (
        control.projected_iterable_bindings
    ):
        if projection == "induction":
            region_value_meta[int(target_id)] = Meta((), "int")
    # Settle strict scalar roles across the complete method before constructing
    # any one region signature.  The same SSA identity can cross several
    # regions; whichever region proves it is an address index or predicate must
    # inform every caller/callee occurrence, independent of traversal order.
    if hierarchy_plan is not None:
        semantic_instructions: list[Instr] = []
        for planned in hierarchy_plan.items:
            if not (
                isinstance(planned, PlanClosure)
                and planned.name.startswith("region_")
            ):
                continue
            planned_instructions = plan_region_to_ssa_instrs(planned, first_free_value_id=plan_value_watermark)
            semantic_instructions.extend(planned_instructions)
            for instruction in planned_instructions:
                for value in (
                    *instruction.args,
                    *((instruction.res,) if instruction.res is not None else ()),
                ):
                    if str(value.dtype) in {"bool", "int", "int32", "int64"}:
                        region_value_meta[int(value.id)] = Meta(
                            tuple(value.shape), str(value.dtype)
                        )
        integer_ids = {
            int(argument.id)
            for instruction in semantic_instructions
            if instruction.op == "GetElementPtr"
            for argument in instruction.args[1:]
        }
        iterable_target_ids = {
            int(target_id)
            for _iterable_id, target_id, _induction
            in control.iterable_bindings
        }
        iterable_target_ids.update({
            int(target_id)
            for _iterable_id, target_id, _induction, projection
            in control.projected_iterable_bindings
            if projection != "induction"
        })
        nested_row_target_ids.update(
            int(instruction.args[0].id)
            for instruction in semantic_instructions
            if instruction.op in {"Indexed", "IndexedStore", "GetElementPtr"}
            and instruction.args
            and int(instruction.args[0].id) in iterable_target_ids
        )
        scalar_use_ids = {
            int(argument.id)
            for instruction in semantic_instructions
            for argument in (
                instruction.args[1:]
                if instruction.op in {
                    "Indexed", "IndexedStore", "GetElementPtr"
                }
                else instruction.args
            )
        }
        indexed_base_ids = {
            int(instruction.args[0].id)
            for instruction in semantic_instructions
            if instruction.op in {"Indexed", "IndexedStore", "GetElementPtr"}
            and instruction.args
        }
        statically_shaped_ids = {
            int(value_id)
            for value_id, meta in region_value_meta.items()
            if tuple(meta.shape or ())
        }
        statically_shaped_ids.update(
            int(value_id)
            for value_id, shape in dict(value_shapes or {}).items()
            if tuple(shape or ())
        )
        variant_projected_target_ids.update(
            target_id
            for target_id in nested_row_target_ids
            if target_id in scalar_use_ids
            and target_id not in statically_shaped_ids
        )
        # The same heterogeneous contract is required when source pursuit has
        # specialized away the owning loop and left its payload as a direct
        # method input.  Its uses, not its spelling, prove the two columns.
        variant_projected_target_ids.update(
            (indexed_base_ids & scalar_use_ids)
            - declared_sequence_ids
            - statically_shaped_ids
        )
        for planned in hierarchy_plan.items:
            if not (
                isinstance(planned, PlanClosure)
                and planned.name.startswith("region_")
            ):
                continue
            region_index = int(planned.name.rsplit("_", 1)[1])
            instructions = plan_region_to_ssa_instrs(planned, first_free_value_id=plan_value_watermark)
            region_array_feed_ids[region_index] = {
                int(instruction.args[0].id)
                for instruction in instructions
                if instruction.op in {"Indexed", "IndexedStore", "GetElementPtr"}
                and instruction.args
                and int(instruction.args[0].id) in variant_projected_target_ids
            }
        nested_iterable_source_ids = {
            int(iterable_id)
            for iterable_id, target_id, _induction
            in control.iterable_bindings
            if int(target_id) in nested_row_target_ids
        }
        selected_nested_sequence_ids.update(
            int(instruction.res.id)
            for instruction in semantic_instructions
            if instruction.op in {"Indexed", "Load"}
            and instruction.res is not None
            and int(instruction.res.id) in nested_iterable_source_ids
        )
        for selected_id in selected_nested_sequence_ids:
            region_value_meta[selected_id] = Meta((), "int")
        integer_ops = {
            "And", "Or", "Xor", "Shl", "Shr", "FloorDiv", "Mod",
            "bitand", "bitor", "bitxor", "shl", "shr", "floordiv", "mod",
        }
        arithmetic_ops = {"Add", "Sub", "Mul", "add", "sub", "mul"}
        changed = True
        while changed:
            changed = False
            for instruction in semantic_instructions:
                result_id = (
                    None if instruction.res is None else int(instruction.res.id)
                )
                argument_ids = tuple(int(value.id) for value in instruction.args)
                if instruction.op in integer_ops:
                    candidates = (*argument_ids, *((result_id,) if result_id is not None else ()))
                elif instruction.op in arithmetic_ops and (
                    (result_id is not None and result_id in integer_ids)
                    or (argument_ids and all(value_id in integer_ids for value_id in argument_ids))
                ):
                    candidates = (*argument_ids, *((result_id,) if result_id is not None else ()))
                else:
                    continue
                for value_id in candidates:
                    if value_id not in integer_ids:
                        integer_ids.add(value_id)
                        changed = True
        for value_id in integer_ids:
            current = region_value_meta.get(value_id)
            region_value_meta[value_id] = Meta(
                () if current is None else tuple(current.shape), "int64"
            )
    table_region_operations: dict[
        int, list[tuple[str, tuple[Any, ...]]]
    ] = {}
    table_region_post_operations: dict[
        int, list[tuple[str, tuple[Any, ...]]]
    ] = {}
    row_load_by_result = {
        int(operation[1]): tuple(operation[1:])
        for operation in sequence_row_operations
        if str(operation[0]) == "load"
    }
    row_store_by_result = {
        int(operation[1]): tuple(operation[1:])
        for operation in sequence_row_operations
        if str(operation[0]) == "store"
    }
    # A fixed-width authored row annotation is authoritative at both sides of
    # a positional update.  Apply its column contract to extracted values and
    # stored leaves before region signatures rebind planner values; otherwise
    # an unknown numeric leaf can retain the planner's conservative float type
    # while its resident column and append helper are correctly integral.
    for operation in sequence_row_operations:
        if str(operation[0]) == "load":
            (
                _kind, result_id, sequence_id, _index_id,
                _index_literal, column, _outer_id,
            ) = operation
            contract = tuple(
                (sequence_column_dtypes or {}).get(int(sequence_id), ())
            )
            if int(column) < len(contract):
                region_value_meta[int(result_id)] = Meta(
                    (), str(contract[int(column)])
                )
        elif str(operation[0]) == "store":
            (
                _kind, _result_id, sequence_id, _index_id,
                _index_literal, leaves, _row_id,
            ) = operation
            contract = tuple(
                (sequence_column_dtypes or {}).get(int(sequence_id), ())
            )
            for column, leaf_id in enumerate(leaves):
                if column < len(contract):
                    region_value_meta[int(leaf_id)] = Meta(
                        (), str(contract[column])
                    )

    sequence_query_result_ids: set[int] = set()

    def collect_sequence_query_results(block: ControlBlock) -> None:
        if isinstance(block, SequenceQueryBlock):
            sequence_query_result_ids.add(int(block.result_value_id))
            sequence_query_result_ids.update(map(int, block.result_alias_ids))
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                collect_sequence_query_results(child)
        elif isinstance(block, ConditionalBlock):
            collect_sequence_query_results(block.body)
            if block.orelse is not None:
                collect_sequence_query_results(block.orelse)
        elif isinstance(block, LoopBlock):
            collect_sequence_query_results(block.body)
        elif isinstance(block, WhileBlock):
            collect_sequence_query_results(block.condition)
            collect_sequence_query_results(block.body)
        elif isinstance(block, CallBlock):
            collect_sequence_query_results(block.callee)

    collect_sequence_query_results(control.root)
    # Seeded with the caller's graph-derived storage aliases (loop-carried
    # versions of in-place mutated arenas chasing to their base storage),
    # so a later loop's reads of "a after loop 1" resolve to a's own
    # buffer instead of materializing a fresh, unconnected formal -- the
    # defect that silently dropped the first of two sequential loops
    # storing to one array (pinned in test_compiled_linalg.py).
    region_value_aliases: dict[int, int] = dict(
        preloaded_value_aliases or {}
    )
    # An in-place pursued call can publish a result identity that is never an
    # operand of a later numerical region: the resident arena itself carries
    # the effect.  Such calls still execute at their authored position.  Use
    # the next retained region as the control anchor, exactly as hierarchical
    # call composition does, instead of waiting for a nonexistent data use.
    if hierarchy_plan is not None and inplace_pack_by_result:
        planned_items = tuple(hierarchy_plan.items)
        consumed_result_ids = {
            int(argument.id)
            for item in planned_items
            if isinstance(item, PlanClosure)
            and item.name.startswith("region_")
            for instruction in plan_region_to_ssa_instrs(item, first_free_value_id=plan_value_watermark)
            for argument in instruction.args
        }
        call_item_positions = {
            int(item.callsite_id): position
            for position, item in enumerate(planned_items)
            if isinstance(item, PlanCall)
        }
        for result_id, (
            sequence_id, width, callee_reference, callsite_id,
        ) in inplace_pack_by_result.items():
            if int(result_id) in consumed_result_ids:
                continue
            position = call_item_positions.get(int(callsite_id))
            if position is None:
                continue
            next_region = next((
                item for item in planned_items[position + 1:]
                if isinstance(item, PlanClosure)
                and item.name.startswith("region_")
            ), None)
            if next_region is None:
                continue
            region_index = int(next_region.name.rsplit("_", 1)[1])
            region_value_aliases[int(result_id)] = int(sequence_id)
            table_region_operations.setdefault(region_index, []).append((
                "pack_bits",
                (
                    sequence_id, sequence_id, width, True,
                    callee_reference, callsite_id,
                ),
            ))
    lookup_by_result = {
        int(result_id): (int(result_id), query_id, int(sequence_id))
        for result_id, query_id, sequence_id in table_lookups
    }
    store_by_effect = {
        int(effect_id): (
            int(effect_id), key_id, int(value_id), int(sequence_id)
        )
        for effect_id, key_id, value_id, sequence_id in table_stores
    }
    region_source_values: list[tuple[int, int]] = []
    handled_table_region_indices: set[int] = set()
    field_read_value_ids = {
        int(value_id)
        for kind, value_id, _slot in field_ops
        if kind == "read"
    }
    authored_output_value_ids = {
        int(history[-1])
        for name in function_outputs
        for history in (
            tuple((identity_table or {}).get(str(name), ())),
        )
        if history
    }
    planned_region_instructions: dict[int, tuple[Instr, ...]] = {}
    resolved_plan_live_value_ids: set[int] = set()
    plan_line_produced: set[int] = set()
    plan_line_consumed: set[int] = set()

    def region_free_value_ids(
        instructions: Sequence[Instr],
    ) -> tuple[int, ...]:
        """Return exact body operands with no local SSA definition.

        Planner captures are scheduling evidence, but the emitted region body
        is the final authority for its callable ABI.  Structural rewrites can
        expose an operand the hierarchy capture set omitted; failing to add it
        creates a function whose instructions reference a value absent from
        its formals.
        """

        produced = {
            int(instruction.res.id)
            for instruction in instructions
            if instruction.res is not None
        }
        return tuple(dict.fromkeys(
            int(argument.id)
            for instruction in instructions
            for argument in instruction.args
            if int(argument.id) not in produced
        ))

    def collect_resolved_plan_dependencies(closure: PlanClosure) -> None:
        """Collect the dependency boundary already resolved by the planner."""

        resolved_plan_live_value_ids.update(int(value_id) for value_id in closure.captures)
        for item in closure.items:
            if isinstance(item, PlanLine):
                plan_line_produced.update(int(v) for v in item.outputs)
                plan_line_consumed.update(int(v) for v in item.inputs)
            if isinstance(item, PlanClosure):
                collect_resolved_plan_dependencies(item)
            elif isinstance(item, PlanCall):
                resolved_plan_live_value_ids.update(
                    int(value_id) for value_id in item.argument_value_ids
                )
                # ``argument_value_ids`` live in the assigned hierarchy
                # namespace. Region SSA is built in the caller's local value
                # namespace, whose exact correlation is the caller side of
                # each PlanCall binding. Preserve both views; do not infer the
                # local feed again from instruction order or terminal values.
                resolved_plan_live_value_ids.update(
                    int(caller_id)
                    for caller_id, _callee_id in item.argument_bindings
                )

    if hierarchy_plan is not None:
        collect_resolved_plan_dependencies(hierarchy_plan)
        # A value some plan line produces that nothing in the hierarchy
        # consumes is either the plan's TERMINAL result or dead planner
        # residue, and the difference is who else is speaking. When ANY
        # output authority is declared -- identity-table function outputs,
        # required ids, per-region output declarations -- declarations are
        # exhaustive, and unconsumed undeclared values are residue to drop
        # (test_cross_region_live_out holds a produced-and-unconsumed value
        # OUT of the boundary for exactly this reason). Only when the plan
        # declares nothing at all do terminals become the result by default:
        # a program whose answer is exported by no one has computed nothing.
        declared_authority = bool(
            (identity_table or {})
            or required_output_value_ids
            or (region_output_value_ids or {})
        )
        if not declared_authority:
            resolved_plan_live_value_ids.update(
                plan_line_produced
                - plan_line_consumed
                - resolved_plan_live_value_ids
            )
        for planned in hierarchy_plan.items:
            if not (
                isinstance(planned, PlanClosure)
                and planned.name.startswith("region_")
            ):
                continue
            planned_index = int(planned.name.rsplit("_", 1)[1])
            planned_instructions = plan_region_to_ssa_instrs(planned, first_free_value_id=plan_value_watermark)
            planned_region_instructions[planned_index] = planned_instructions

    if hierarchy_plan is not None:
        for region in hierarchy_plan.items:
            if not (
                isinstance(region, PlanClosure)
                and region.name.startswith("region_")
            ):
                continue
            region_index = int(region.name.split("_", 1)[1])
            if region_index not in set(map(int, control.region_indices)):
                # Projection removed this region's only lexical marker. This
                # is how an external-reference occurrence replaces the old
                # capture-time numerical imitation of the same call; keeping
                # the detached region would execute both meanings.
                continue
            if region_index in region_callees:
                continue
            # Namespace regions by their owning method so two methods that each
            # carve a ``region_0`` do not collide in one shared library, and so
            # the control call the lowering emits already targets this symbol.
            region_name = f"{control_name}__planned_region_{region_index}"
            instructions = list(plan_region_to_ssa_instrs(region))
            # A resident generator query (currently ``sum(1 for ...)`` or
            # ``next(..., default)``) is emitted in lexical control beside its
            # producer loop. Remove only that replaced operator from the
            # numerical region; downstream arithmetic/compare instructions
            # remain and consume the query's exact result identity.
            instructions = [
                instruction for instruction in instructions
                if instruction.res is None
                or int(instruction.res.id) not in sequence_query_result_ids
            ]
            row_loads = tuple(
                row_load_by_result[int(instruction.res.id)]
                for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id) in row_load_by_result
            )
            row_stores = tuple(
                row_store_by_result[int(instruction.res.id)]
                for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id) in row_store_by_result
            )
            if row_loads or row_stores:
                removed_results = {
                    int(operation[0]) for operation in (*row_loads, *row_stores)
                }
                instructions = [
                    instruction for instruction in instructions
                    if instruction.res is None
                    or int(instruction.res.id) not in removed_results
                ]
                while True:
                    consumed = {
                        int(argument.id)
                        for instruction in instructions
                        for argument in instruction.args
                    }
                    dead_constants = {
                        int(instruction.res.id)
                        for instruction in instructions
                        if instruction.op == "Const"
                        and instruction.res is not None
                        and int(instruction.res.id) not in consumed
                    }
                    if not dead_constants:
                        break
                    instructions = [
                        instruction for instruction in instructions
                        if instruction.res is None
                        or int(instruction.res.id) not in dead_constants
                    ]
                for operation in row_loads:
                    result_id = int(operation[0])
                    table_region_operations.setdefault(
                        region_index, []
                    ).append((
                        "row_load_capture" if instructions else "row_load",
                        operation,
                    ))
                for operation in row_stores:
                    result_id, sequence_id = map(int, operation[:2])
                    region_value_aliases[result_id] = sequence_id
                    table_region_operations.setdefault(
                        region_index, []
                    ).append(("row_store", operation))
            # Instance-field reads already have an exact record slot contract.
            # Keeping their source-level GetAttr instruction inside a flat
            # region throws that information away and asks every backend to
            # rediscover Python object semantics. Remove that instruction here:
            # a mixed region receives the loaded field value as a capture, and
            # a field-only region disappears. `_inject_field_slot_access`
            # inserts the corresponding GEP/Load at the first region call (or
            # at Ret for a getter), so the object has become ordinary storage
            # before target emission.
            resolved_field_reads = {
                int(instruction.res.id)
                for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id) in field_read_value_ids
                and str(instruction.op).casefold() == "getattr"
            }
            if resolved_field_reads:
                instructions = [
                    instruction
                    for instruction in instructions
                    if not (
                        instruction.res is not None
                        and int(instruction.res.id) in resolved_field_reads
                        and str(instruction.op).casefold() == "getattr"
                    )
                ]
            # Attribute lookup itself is pure. Planner regions historically
            # retained method/property lookup nodes as terminal outputs even
            # when a typed PlanCall, structural helper, or tensor operation
            # already carried the actual effect. Prune only terminal lookups
            # that neither cross a region boundary nor implement an authored
            # return. Iterate so a dead ``obj.__dict__.get`` chain collapses
            # from the outside inward.
            while True:
                locally_consumed = {
                    int(argument.id)
                    for instruction in instructions
                    for argument in instruction.args
                }
                removable = {
                    int(instruction.res.id)
                    for instruction in instructions
                    if instruction.res is not None
                    and str(instruction.op).casefold() == "getattr"
                    and int(instruction.res.id) not in locally_consumed
                    and int(instruction.res.id) not in resolved_plan_live_value_ids
                    and int(instruction.res.id) not in authored_output_value_ids
                }
                if not removable:
                    break
                instructions = [
                    instruction
                    for instruction in instructions
                    if instruction.res is None
                    or int(instruction.res.id) not in removable
                ]
            effective_captures = tuple(dict.fromkeys((
                *map(int, region.captures),
                *region_free_value_ids(instructions),
                *(
                    int(operation[0])
                    for operation in row_loads
                    if instructions
                ),
                *(
                    int(argument.id)
                    for instruction in instructions
                    for argument in instruction.args
                    if int(argument.id) in sequence_query_result_ids
                ),
                *(
                    int(argument.id)
                    for instruction in instructions
                    for argument in instruction.args
                    if int(argument.id) in resolved_field_reads
                ),
            )))
            produced_after_field_resolution = {
                int(instruction.res.id)
                for instruction in instructions
                if instruction.res is not None
            }
            consumed_after_field_resolution = {
                int(argument.id)
                for instruction in instructions
                for argument in instruction.args
            }
            region_inout_ids = tuple(
                value_id
                for value_id in map(int, record_field_write_value_ids)
                if value_id in produced_after_field_resolution
            )
            effective_captures = tuple(dict.fromkeys((
                *effective_captures,
                *region_inout_ids,
            )))
            effective_captures = tuple(
                value_id
                for value_id in effective_captures
                if (
                    value_id in region_inout_ids
                    or value_id in region_free_value_ids(instructions)
                )
            )
            if not instructions:
                table_region_operations.setdefault(region_index, []).append((
                    "structural_consumed", (),
                ))
                continue
            if any(
                instruction.res is not None
                and int(instruction.res.id) in (
                    append_fill_source_result_ids
                    | append_slice_source_result_ids
                )
                for instruction in instructions
            ):
                table_region_operations.setdefault(region_index, []).append((
                    "structural_consumed", (),
                ))
                continue
            if any(
                instruction.res is not None
                and int(instruction.res.id) in bit_pack_consumed_ids
                for instruction in instructions
            ):
                table_region_operations.setdefault(region_index, []).append((
                    "structural_consumed", (),
                ))
                continue
            fill_instructions = tuple(
                instruction for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id) in sequence_fills
                and instruction.op in {"Mul", "Mult", "mul"}
            )
            if fill_instructions:
                removed_ids = {
                    int(instruction.res.id)
                    for instruction in fill_instructions
                }
                # The list literal feeding replication is structural input to
                # the fill helper; remove its now-dead Const from the region.
                fill_operand_ids = {
                    int(argument.id)
                    for instruction in fill_instructions
                    for argument in instruction.args
                }
                instructions = [
                    instruction for instruction in instructions
                    if instruction not in fill_instructions
                    and not (
                        instruction.op == "Const"
                        and instruction.res is not None
                        and int(instruction.res.id) in fill_operand_ids
                        and isinstance(instruction.attributes.get("value"), (list, tuple))
                    )
                ]
                for result_id in removed_ids:
                    table_region_post_operations.setdefault(
                        region_index, []
                    ).append(("fill", sequence_fills[result_id]))
            augment_instructions = tuple(
                instruction
                for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id) in sequence_augment_by_result
                and instruction.op in {"Add", "add"}
            )
            if augment_instructions:
                for instruction in augment_instructions:
                    result_id = int(instruction.res.id)
                    destination_id, source_id = sequence_augment_by_result[
                        result_id
                    ]
                    region_value_aliases[result_id] = destination_id
                    table_region_post_operations.setdefault(
                        region_index, []
                    ).append((
                        "extend",
                        (result_id, destination_id, source_id),
                    ))
                instructions = [
                    instruction for instruction in instructions
                    if instruction not in augment_instructions
                    and not (
                        instruction.op == "Const"
                        and instruction.res is not None
                        and not any(
                            int(instruction.res.id) == int(argument.id)
                            for remaining in instructions
                            if remaining not in augment_instructions
                            for argument in remaining.args
                        )
                    )
                ]
            concat_instructions = tuple(
                instruction
                for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id) in sequence_concat_by_result
                and instruction.op in {"Add", "add"}
            )
            if concat_instructions:
                for instruction in concat_instructions:
                    result_id = int(instruction.res.id)
                    (
                        lhs_id, rhs_id, _kind, lhs_scalar, rhs_scalar,
                    ) = sequence_concat_by_result[
                        result_id
                    ]
                    region_value_aliases[result_id] = result_id
                    scheduled = table_region_operations.setdefault(
                        region_index, []
                    )
                    scheduled.append(("reset", (result_id,)))
                    scheduled.append((
                        "append_scalar" if lhs_scalar is not None
                        else "append_slice",
                        (
                            (result_id, lhs_scalar)
                            if lhs_scalar is not None
                            else (result_id, lhs_id, None, None)
                        ),
                    ))
                    scheduled.append((
                        "append_scalar" if rhs_scalar is not None
                        else "append_slice",
                        (
                            (result_id, rhs_scalar)
                            if rhs_scalar is not None
                            else (result_id, rhs_id, None, None)
                        ),
                    ))
                instructions = [
                    instruction for instruction in instructions
                    if instruction not in concat_instructions
                ]
                if not instructions:
                    table_region_operations.setdefault(
                        region_index, []
                    ).append(("structural_consumed", ()))
                    continue
            append_fill_instructions = tuple(
                instruction
                for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id)
                in sequence_append_fill_by_result
                and instruction.op in {"Add", "add"}
            )
            if append_fill_instructions:
                for instruction in append_fill_instructions:
                    result_id = int(instruction.res.id)
                    destination_id, literal, count_id = (
                        sequence_append_fill_by_result[result_id]
                    )
                    region_value_aliases[result_id] = destination_id
                    table_region_post_operations.setdefault(
                        region_index, []
                    ).append((
                        "append_fill",
                        (destination_id, literal, count_id),
                    ))
                instructions = [
                    instruction for instruction in instructions
                    if instruction not in append_fill_instructions
                ]
            append_slice_instructions = tuple(
                instruction
                for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id) in sequence_append_slice_by_result
                and instruction.op in {"Add", "add"}
            )
            if append_slice_instructions:
                for instruction in append_slice_instructions:
                    result_id = int(instruction.res.id)
                    destination_id, source_id, lower_id, upper_id = (
                        sequence_append_slice_by_result[result_id]
                    )
                    region_value_aliases[result_id] = destination_id
                    table_region_post_operations.setdefault(
                        region_index, []
                    ).append((
                        "append_slice",
                        (destination_id, source_id, lower_id, upper_id),
                    ))
                instructions = [
                    instruction for instruction in instructions
                    if instruction not in append_slice_instructions
                ]
            prepend_instructions = tuple(
                instruction for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id) in prepend_by_result
                and instruction.op in {"Add", "add"}
            )
            if prepend_instructions:
                for instruction in prepend_instructions:
                    result_id = int(instruction.res.id)
                    sequence_id, value_id = prepend_by_result[result_id]
                    region_value_aliases[result_id] = sequence_id
                    table_region_post_operations.setdefault(
                        region_index, []
                    ).append(("prepend", (sequence_id, value_id)))
                instructions = [
                    instruction for instruction in instructions
                    if instruction not in prepend_instructions
                ]
            packed_instructions = tuple(
                instruction for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id) in packed_by_concat
                and instruction.op in {"Add", "add"}
            )
            if packed_instructions:
                for instruction in packed_instructions:
                    result_id = int(instruction.res.id)
                    (
                        destination_id, source_id, prefix_id,
                        packed_callsite_id,
                    ) = packed_by_concat[
                        result_id
                    ]
                    region_value_aliases[result_id] = destination_id
                instructions = [
                    instruction for instruction in instructions
                    if instruction not in packed_instructions
                ]
            packed_store_instructions = tuple(
                instruction for instruction in instructions
                if instruction.res is not None
                and int(instruction.res.id) in packed_by_store
            )
            if packed_store_instructions:
                for instruction in packed_store_instructions:
                    store_result_id = int(instruction.res.id)
                    (
                        destination_id, source_id, prefix_id,
                        packed_callsite_id,
                    ) = packed_by_store[
                        store_result_id
                    ]
                    # Preserve authored call order at the splice boundary:
                    # bitmap-pack destination, then prepend the prefix and
                    # packed mapping words, then consume the slice store.
                    prior_pack = next((
                        inplace_pack_by_result[result_id]
                        for result_id in sorted(inplace_pack_by_result)
                        if result_id in {int(value.id) for value in instruction.args}
                    ), None)
                    if prior_pack is not None:
                        (
                            sequence_id, width, callee_reference,
                            pack_callsite_id,
                        ) = prior_pack
                        region_value_aliases[next(
                            result_id for result_id, contract
                            in inplace_pack_by_result.items()
                            if contract == prior_pack
                            and result_id in {
                                int(value.id) for value in instruction.args
                            }
                        )] = sequence_id
                        table_region_operations.setdefault(
                            region_index, []
                        ).append((
                            "pack_bits",
                            (
                                sequence_id, sequence_id, width, True,
                                callee_reference, pack_callsite_id,
                            ),
                        ))
                    table_region_operations.setdefault(
                        region_index, []
                    ).append((
                        "prepend_packed_bytes",
                        (
                            destination_id, source_id, prefix_id, 4,
                            packed_callsite_id,
                        ),
                    ))
                table_region_operations.setdefault(region_index, []).append((
                    "structural_consumed", (),
                ))
                continue
            if any(
                instruction.res is not None
                and int(instruction.res.id) in prepend_store_result_ids
                for instruction in instructions
            ):
                # The following zero-width slice store is the authored splice;
                # prepend has already performed it in resident memory.
                table_region_operations.setdefault(region_index, []).append((
                    "structural_consumed", (),
                ))
                continue
            # A pursued structural bit-pack call publishes the same resident
            # arena under its call result ID.  Schedule the helper at the first
            # region that consumes that result, before that region executes.
            consumed_pack_results = tuple(sorted({
                int(argument.id)
                for instruction in instructions
                for argument in instruction.args
                if int(argument.id) in inplace_pack_by_result
            }))
            for result_id in consumed_pack_results:
                (
                    sequence_id, width, callee_reference,
                    pack_callsite_id,
                ) = inplace_pack_by_result[
                    result_id
                ]
                region_value_aliases[result_id] = sequence_id
                table_region_operations.setdefault(region_index, []).append((
                    "pack_bits",
                    (
                        sequence_id, sequence_id, width, True,
                        callee_reference, pack_callsite_id,
                    ),
                ))
            for instruction in instructions:
                if (
                    instruction.op == "IndexedStore"
                    and instruction.res is not None
                    and instruction.args
                ):
                    # IndexedStore versions resident memory; it does not return
                    # a new scalar payload.  Preserve the authored result ID as
                    # an alias of the base arena across subsequent regions.
                    region_value_aliases[int(instruction.res.id)] = int(
                        instruction.args[0].id
                    )
            region_source_values.extend(
                (int(instruction.res.id), region_index)
                for instruction in instructions
                if instruction.res is not None
            )
            table_index_instructions = tuple(
                instruction
                for instruction in instructions
                if instruction.op in {"Indexed", "IndexedStore"}
                and instruction.res is not None
                and (
                    int(instruction.res.id) in table_lookup_result_ids
                    or int(instruction.res.id) in table_store_effect_ids
                )
            )
            if table_index_instructions:
                if (
                    len(table_index_instructions) == len(instructions) == 1
                    and table_index_instructions[0].op == "Indexed"
                    and table_index_instructions[0].res is not None
                    and int(table_index_instructions[0].res.id)
                    in table_lookup_result_ids
                ):
                    handled_table_region_indices.add(region_index)
                    table_region_operations.setdefault(region_index, []).append((
                        "lookup",
                        lookup_by_result[int(table_index_instructions[0].res.id)],
                    ))
                    continue
                if (
                    len(table_index_instructions) == len(instructions) == 1
                    and table_index_instructions[0].op == "IndexedStore"
                    and table_index_instructions[0].res is not None
                    and int(table_index_instructions[0].res.id)
                    in table_store_effect_ids
                ):
                    handled_table_region_indices.add(region_index)
                    table_region_operations.setdefault(region_index, []).append((
                        "store",
                        store_by_effect[int(table_index_instructions[0].res.id)],
                    ))
                    continue
                table_ids = {
                    int(instruction.res.id)
                    for instruction in table_index_instructions
                    if instruction.res is not None
                }
                remaining = [
                    instruction for instruction in instructions
                    if instruction not in table_index_instructions
                ]
                consumed_table_ids = {
                    int(argument.id)
                    for instruction in remaining
                    for argument in instruction.args
                    if int(argument.id) in table_ids
                }
                if consumed_table_ids:
                    unsupported_consumed = tuple(
                        instruction
                        for instruction in table_index_instructions
                        if (
                            int(instruction.res.id) in consumed_table_ids
                            and (
                                instruction.op != "Indexed"
                                or int(instruction.res.id) not in lookup_by_result
                            )
                        )
                    )
                    if unsupported_consumed:
                        for instruction in unsupported_consumed:
                            shortfalls.append(SSALoweringShortfall(
                                "ssa-table",
                                instruction.op,
                                f"{control_name}.region_{region_index}",
                                "keyed table effect is consumed as a scalar but is not a lookup",
                            ))
                        continue
                    for result_id in sorted(consumed_table_ids):
                        table_region_operations.setdefault(
                            region_index, []
                        ).append((
                            "lookup_capture", lookup_by_result[result_id]
                        ))
                    effective_captures = tuple(dict.fromkeys((
                        *effective_captures,
                        *sorted(consumed_table_ids),
                    )))
                instructions = remaining
                for instruction in table_index_instructions:
                    effect_id = int(instruction.res.id)
                    if effect_id in consumed_table_ids:
                        continue
                    if instruction.op == "Indexed":
                        table_region_post_operations.setdefault(
                            region_index, []
                        ).append(("lookup", lookup_by_result[effect_id]))
                    else:
                        table_region_post_operations.setdefault(
                            region_index, []
                        ).append(("store", store_by_effect[effect_id]))
            instruction_values = {
                int(value.id): value
                for instruction in instructions
                for value in (
                    *instruction.args,
                    *((instruction.res,) if instruction.res is not None else ()),
                )
            }
            # A variant payload has parallel scalar and child-row columns.
            # Its authored ID remains scalar in regions that use it as a value
            # or index; only regions that dereference it receive the row-base
            # address selected by the control caller.
            for value_id in variant_projected_target_ids:
                if value_id in region_array_feed_ids.get(region_index, set()):
                    region_value_meta[value_id] = Meta((), "float64")
                elif value_id in instruction_values:
                    region_value_meta[value_id] = Meta((), "float64")
            region_values = {}
            for value_id, shape, dtype in region.value_shapes:
                authoritative = region_value_meta.get(int(value_id))
                semantic = instruction_values.get(int(value_id))
                region_values[int(value_id)] = SSAValue(
                    int(value_id),
                    dtype=(
                        str(authoritative.dtype)
                        if authoritative is not None else str(dtype)
                        if semantic is None or semantic.dtype is None
                        else str(semantic.dtype)
                    ),
                    shape=(
                        # One storage identity may enter different planned
                        # regions through different reshape/view contracts.
                        # The instruction occurrence is the exact local view;
                        # the id-keyed region-wide metadata is only a fallback
                        # for otherwise shapeless occurrences.
                        tuple(semantic.shape)
                        if semantic is not None and semantic.shape
                        else tuple(map(int, authoritative.shape))
                        if authoritative is not None
                        else tuple(map(int, shape))
                    ),
                )
            for value_id, shape, dtype in region.value_shapes:
                semantic = instruction_values.get(int(value_id))
                region_value_meta.setdefault(
                    int(value_id),
                    Meta(
                        shape=(
                            tuple(semantic.shape)
                            if semantic is not None and semantic.shape
                            else tuple(map(int, shape))
                        ),
                        dtype=(
                            str(semantic.dtype)
                            if semantic is not None and semantic.dtype is not None
                            else str(dtype)
                        ),
                    ),
                )

            def typed_region_value(value_id: int) -> SSAValue:
                return region_values.get(
                    int(value_id),
                    instruction_values.get(
                        int(value_id), SSAValue(int(value_id))
                    ),
                )

            # One SSA identity has one storage/dtype contract, but may have
            # several shaped views. Retype each occurrence independently so
            # reshape metadata is not collapsed by the id-keyed canonical
            # table. Target inference must see the authored view at each op.
            for instruction in instructions:
                instruction.args = [
                    _typed_region_occurrence(
                        value,
                        typed_region_value(value.id),
                        region_value_meta.get(int(value.id)),
                    )
                    for value in instruction.args
                ]
                if instruction.res is not None:
                    instruction.res = _typed_region_occurrence(
                        instruction.res,
                        typed_region_value(instruction.res.id),
                        region_value_meta.get(int(instruction.res.id)),
                    )

            # An aggregate formal IS the tuple of its member formals: its
            # ``Tuple`` constructor has no physical value.  Every projection
            # of it is aliased to a member, and a starred call argument
            # (``step(inputs, *tire_constants)``) is expanded by the linker
            # into those members, so a plan-level reference to the tuple id
            # is never a physical use.  A backend has no spelling for the
            # constructor and must not be asked for one: keep it only while
            # an instruction in this region consumes it, or it is itself an
            # authored function output.
            consumed_in_region = {
                int(argument.id)
                for instr in instructions
                for argument in instr.args
            }
            instructions = [
                instr for instr in instructions
                if not (
                    str(instr.op) in {"Tuple", "tuple"}
                    and instr.res is not None
                    and int(instr.res.id) not in consumed_in_region
                    and int(instr.res.id) not in authored_output_value_ids
                )
            ]
            produced = {
                int(instr.res.id)
                for instr in instructions
                if instr.res is not None
            }
            consumed = {
                int(argument.id)
                for instr in instructions
                for argument in instr.args
            }
            effective_captures = tuple(
                value_id
                for value_id in dict.fromkeys((
                    *effective_captures,
                    *region_free_value_ids(instructions),
                ))
                if (
                    value_id in region_inout_ids
                    or value_id in region_free_value_ids(instructions)
                )
            )
            # The resolved hierarchy plan owns the dependency calculation.
            # A region exports exactly the values crossing that boundary (plus
            # authored function results), never locally terminal temporaries.
            required_outputs = (
                resolved_plan_live_value_ids
                | set(control_dependency_value_ids(control))
                | authored_output_value_ids
                | set(map(int, required_output_value_ids))
                | set(map(
                    int,
                    (region_output_value_ids or {}).get(region_index, ()),
                ))
                | set(map(int, record_field_write_value_ids))
            )
            def resident_root(value_id: int) -> int:
                current = int(value_id)
                seen: set[int] = set()
                while (
                    current in region_value_aliases
                    and current not in seen
                ):
                    seen.add(current)
                    current = int(region_value_aliases[current])
                return current

            # An IndexedStore versions resident memory: every version id in
            # ``region_value_aliases`` names the SAME storage as its root
            # arena.  A requirement on a version is therefore a requirement
            # on that storage, so the produced root is published; and the
            # required version is published AS ITSELF, because the version id
            # is the identity every later stage keys on (the identity ledger,
            # the structural-output recovery, ``emit_outputs`` and the call
            # linker all name a returned store chain by its final version).
            # This is exactly what the repository-call lowering of the same
            # store already does (``index_assign`` results are published next
            # to their arena).  Intermediate versions nothing requires stay
            # private.  A version whose root is a CAPTURE (a parameter or an
            # earlier region's arena written in place here) is not published:
            # the write is already visible through that storage, and a copied
            # output slot would turn an in/out write into a second value.
            # Dropping every aliased id silently removed each authored output
            # built by in-place slice stores (``surface_velocity = point *
            # 0.0; surface_velocity[..., 0] = ...``) from the region and then
            # from the function's Ret.
            required_roots = {
                resident_root(value_id) for value_id in required_outputs
            }
            outputs = tuple(sorted(
                value_id for value_id in produced
                if (
                    value_id in required_outputs
                    and (
                        value_id not in region_value_aliases
                        or resident_root(value_id) in produced
                    )
                )
                or (
                    value_id in required_roots
                    and value_id not in region_value_aliases
                )
            ))
            # The region's formal parameters are its captures only. Its outputs
            # are declared as ``intent(out)`` dummies by the target from the
            # ``outputs`` map (returned below as ``section_outputs``), exactly as
            # the fused numerical region path does -- never by placing them in
            # ``args``, which would misread an output as an in/out alias.
            def region_argument(value_id: int) -> SSAValue:
                canonical = typed_region_value(value_id)
                occurrences = tuple(
                    argument
                    for instruction in instructions
                    for argument in instruction.args
                    if int(argument.id) == int(value_id)
                    and tuple(argument.shape or ())
                )
                contracts = {
                    (tuple(value.shape), str(value.dtype or ""))
                    for value in occurrences
                }
                # A sole concrete use is the exact ordered region view. When
                # several reshapes share the storage id, keep the canonical
                # formal neutral; the individual op occurrences above retain
                # their own views.
                if len(contracts) == 1:
                    return occurrences[0]
                return canonical

            arguments = [region_argument(vid) for vid in effective_captures]
            region_function = Function(
                region_name,
                arguments,
                {"entry": BasicBlock("entry", instructions)},
            )
            # A planned region is a compile-complementary source integral,
            # not an anonymous backend kernel.  Retain its deterministic
            # structural address so a bounded project compile can publish a
            # complete region even when the enclosing authored method still
            # needs Python object semantics.  The chain is deliberately
            # reversible text, not a digest or a process-local numeric ID.
            region_function.metadata["source_region_integral"] = {
                "schema": "turing.source-region-integral.v1",
                "owner": str(control_name),
                "plan_name": str(region.name),
                "region_index": int(region_index),
                "closure_id": int(region.closure_id),
                "identity_token_chain": (
                    "source-region",
                    str(control_name),
                    f"closure:{int(region.closure_id)}",
                    str(region.name),
                ),
                "capture_value_ids": tuple(map(int, effective_captures)),
                "output_value_ids": tuple(map(int, outputs)),
            }
            region_function.metadata["scalar_variant_argument_ids"] = tuple(
                sorted(
                    value_id
                    for value_id in variant_projected_target_ids
                    if value_id in effective_captures
                    and value_id not in region_array_feed_ids.get(
                        region_index, set()
                    )
                )
            )
            functions[region_name] = region_function
            region_callees[region_index] = region_name
            region_signatures[region_index] = (
                tuple(int(vid) for vid in effective_captures),
                outputs,
            )
            region_feed_meta[region_index] = tuple(
                Meta(
                    tuple(argument.shape),
                    str(argument.dtype or "unknown"),
                    argument.device,
                )
                for argument in arguments
            )
            section_outputs[region_name] = tuple(
                typed_region_value(vid) for vid in outputs
            )
            # Do NOT gate region ops on the repository ``Handler`` enum here: that
            # is the LLVM/repository vocabulary, but this path emits through the
            # selected target (Fortran), whose op set is broader -- e.g. it
            # renders ``equal`` as ``(a == b)`` though ``Handler`` only knows
            # ``Eq``. The target's own emit reports any op it genuinely cannot
            # express, with a message accurate to that target.
    table_epilogue_operations: list[tuple[str, tuple[Any, ...]]] = []
    iterable_source_ids = {
        int(binding[0]) for binding in control.iterable_bindings
    }
    for deletion in table_deletions:
        effect_id = int(deletion[0])
        preceding = [
            (value_id, region_index)
            for value_id, region_index in region_source_values
            if value_id < effect_id
        ]
        following = [
            (value_id, region_index)
            for value_id, region_index in region_source_values
            if value_id > effect_id
        ]
        if deletion[2] is None and preceding:
            _value_id, region_index = max(preceding)
            table_region_operations.setdefault(region_index, []).append(("delete", deletion))
        elif preceding and max(preceding)[0] in iterable_source_ids:
            _value_id, region_index = max(preceding)
            table_region_operations.setdefault(region_index, []).append(("delete", deletion))
        elif following:
            _value_id, region_index = min(following)
            table_region_operations.setdefault(region_index, []).insert(0, ("delete", deletion))
        else:
            table_epilogue_operations.append(("delete", deletion))
    for destination_id, source_id, width_id, _consumed_ids in sequence_bit_packs:
        table_epilogue_operations.append((
            "pack_bits", (
                int(destination_id), int(source_id), int(width_id)
            ),
        ))
    # The control lowering mints synthetic values (aggregate handles, index
    # constants) for the region-call convention. They must not reuse a graph
    # value id, or a synthetic const collides with a field value the injection
    # below references by that same id. Start them above every graph id in play.
    graph_value_ids = [0]
    for history in (identity_table or {}).values():
        graph_value_ids.extend(int(value_id) for value_id in history)
    for feeds, outputs in region_signatures.values():
        graph_value_ids.extend(int(value_id) for value_id in feeds)
        graph_value_ids.extend(int(value_id) for value_id in outputs)
    graph_value_ids.extend(int(value_id) for _kind, value_id, _slot in field_ops)
    if self_value_id is not None:
        graph_value_ids.append(int(self_value_id))
    for value_id in set(value_dtypes or {}) | set(value_shapes or {}):
        value_id = int(value_id)
        existing = region_value_meta.get(value_id)
        region_value_meta[value_id] = Meta(
            tuple((value_shapes or {}).get(
                value_id,
                tuple(existing.shape or ()) if existing is not None else (),
            )),
            str((value_dtypes or {}).get(
                value_id,
                existing.dtype if existing is not None else "float64",
            )),
            existing.device if existing is not None else None,
        )
    # Keep the hierarchy's pre-rewrite dataflow as scheduling authority.  A
    # structural region may disappear entirely into resident-sequence
    # operations and therefore never receive a numerical ``region_signature``;
    # its original instructions still say exactly which identities it consumes
    # and publishes.  Reading arbitrary integer leaves out of operation tuples
    # would conflate IDs with widths, slots, literals, and callsite numbers.
    region_dependency_signatures = {}
    for region_index, instructions in planned_region_instructions.items():
        produced = tuple(dict.fromkeys(
            int(instruction.res.id)
            for instruction in instructions
            if instruction.res is not None
        ))
        produced_set = set(produced)
        consumed = tuple(dict.fromkeys(
            int(argument.id)
            for instruction in instructions
            for argument in instruction.args
            if int(argument.id) not in produced_set
        ))
        region_dependency_signatures[int(region_index)] = (consumed, produced)

    region_scheduled_lookup_ids = {
        int(operation[0])
        for operations in (
            *table_region_operations.values(),
            *table_region_post_operations.values(),
        )
        for kind, operation in operations
        if str(kind) in {"lookup", "lookup_capture"}
    }
    control, lexical_table_lookup_ids, lookup_refusals = (
        _install_loop_owned_table_queries(
            control,
            table_lookups,
            table_lookup_loop_owners or {},
            excluded_result_ids=region_scheduled_lookup_ids,
            globally_mutated_sequence_ids=(
                int(sequence_id)
                for _effect, _key, _value, sequence_id in table_stores
            ),
        )
    )
    shortfalls.extend(
        SSALoweringShortfall(
            "ssa-table", "lexical-lookup", control_name,
            f"lookup {result_id} cannot be placed: {reason}",
        )
        for result_id, reason in lookup_refusals
    )
    control, plan_callsite_bindings = _schedule_loop_callsites(
        control,
        hierarchy_plan,
        region_signatures,
        region_dependency_signatures,
    )
    control_function, control_shortfalls = lower_control_program_to_ssa(
        control,
        function_name=control_name,
        first_value_id=max(graph_value_ids) + 1,
        region_callees=region_callees,
        region_signatures=region_signatures,
        region_feed_meta=region_feed_meta,
        region_value_meta=region_value_meta,
        plan_callsite_bindings=plan_callsite_bindings,
        value_aliases=region_value_aliases,
        inout_value_ids=tuple(map(int, record_field_write_value_ids)),
        output_value_ids=tuple(map(int, required_output_value_ids)),
        named_output_histories={
            str(name): tuple(map(int, (identity_table or {}).get(name, ())))
            for name in function_outputs
        },
        value_name_histories=identity_table,
        parameter_names=function_parameters,
        sequence_initializations=sequence_initializations,
        sequence_declarations=sequence_declarations,
        sequence_column_dtypes=sequence_column_dtypes,
        sequence_record_identities=sequence_record_identities,
        source_sequence_ids=source_sequence_ids,
        sequence_memberships=sequence_memberships,
        table_lookups=table_lookups,
        lexical_table_lookup_result_ids=lexical_table_lookup_ids,
        table_lookup_defaults=table_lookup_defaults,
        table_stores=table_stores,
        table_deletions=table_deletions,
        retained_sequence_ids=retained_sequence_ids,
        nested_sequence_ids=nested_sequence_ids,
        joined_sequence_ids=joined_sequence_ids,
        joined_singleton_values=joined_singleton_values,
        nested_row_target_ids=tuple(sorted(
            nested_row_target_ids
        )),
        selected_nested_sequence_ids=tuple(sorted(
            selected_nested_sequence_ids
        )),
        variant_projected_target_ids=tuple(sorted(
            variant_projected_target_ids
        )),
        region_array_feed_ids={
            region: tuple(sorted(value_ids))
            for region, value_ids in region_array_feed_ids.items()
            if value_ids
        },
        nested_row_projections=nested_row_projections,
        sequence_length_values=sequence_length_values,
        resolved_sequence_schemas=resolved_sequence_schemas,
        table_region_operations={
            region: tuple(operations)
            for region, operations in table_region_operations.items()
        },
        table_region_post_operations={
            region: tuple(operations)
            for region, operations in table_region_post_operations.items()
        },
        table_epilogue_operations=tuple(table_epilogue_operations),
    )
    control_function = _materialize_control_constants(
        control_function,
        constant_values or {},
        value_dtypes=value_dtypes or {},
    )
    sequence_helpers, sequence_tables = _sequence_artifacts_from_control(
        control_function
    )
    control_sequence_table = sequence_tables.get(control_function.name)
    control_sequences = (
        dict(control_sequence_table.sequences)
        if control_sequence_table is not None else {}
    )
    sequence_field_slots: dict[int, int] = {}
    for _kind, value_id, slot in field_ops:
        if int(value_id) in control_sequences:
            sequence_field_slots[int(slot)] = int(value_id)
    for alias_slot, target_slot in field_aliases:
        if int(target_slot) in sequence_field_slots:
            sequence_field_slots[int(alias_slot)] = sequence_field_slots[
                int(target_slot)
            ]
    # Only fields this method actually touches belong to its scalar receiver
    # view. Sequence slots travel through their descriptors, while unrelated
    # class fields are neither dependencies nor native ABI parameters.
    scalar_slots = tuple(sorted({
        int(slot)
        for _kind, _value_id, slot in field_ops
        if int(slot) not in sequence_field_slots
    }))
    compact_slot = {slot: index for index, slot in enumerate(scalar_slots)}
    scalar_field_ops = tuple(
        (kind, value_id, compact_slot[slot])
        for kind, value_id, slot in field_ops
        if slot in compact_slot
    )
    declared_field_dtypes = {
        str(name): str(dtype)
        for name, dtype in dict(record_field_dtypes or {}).items()
        if dtype is not None
    }
    scalar_dtypes = {
        declared_field_dtypes[name]
        for slot, name in enumerate(field_names)
        if slot in compact_slot and name in declared_field_dtypes
    }
    undeclared_scalar_fields = tuple(
        name for slot, name in enumerate(field_names)
        if slot in compact_slot and name not in declared_field_dtypes
    )
    if scalar_slots and declared_field_dtypes and undeclared_scalar_fields:
        raise ValueError(
            "record scalar slot ABI lacks dtype for fields "
            + ", ".join(undeclared_scalar_fields)
        )
    receiver_scalar_dtype = (
        sorted(scalar_dtypes)[0] if scalar_dtypes else "float64"
    )
    scalar_slot_dtypes = {
        compact_slot[old_slot]: declared_field_dtypes.get(name, receiver_scalar_dtype)
        for old_slot, name in enumerate(field_names)
        if old_slot in compact_slot
    }
    scalar_field_locations: dict[int, tuple[int, int, str]] = {}
    # ``self`` is a compile-time record correlation, not an opaque runtime
    # object.  When every field is already represented by explicit sequence
    # arenas, the receiver has no remaining scalar storage and must not become
    # an otherwise-unused ABI argument.  Method-call linking correlates the
    # authored receiver identity to the record descriptor and passes the field
    # arenas themselves.
    if (
        self_value_id is not None
        and record_identity is not None
        and not scalar_slots
        and not any(
            int(argument.id) == int(self_value_id)
            for block in control_function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        )
    ):
        control_function.args = [
            argument for argument in control_function.args
            if int(argument.id) != int(self_value_id)
        ]
    if self_value_id is not None and scalar_slots:
        non_self_param_ids = tuple(
            int((identity_table or {}).get(name, (None,))[-1])
            for name in function_parameters
            if name != "self" and (identity_table or {}).get(name)
        )
        output_value_ids = tuple(
            int((identity_table or {}).get(name, (None,))[-1])
            for name in function_outputs
            if (identity_table or {}).get(name)
        )
        control_function, scalar_field_locations = _inject_field_slot_access(
            control_function,
            self_value_id=int(self_value_id),
            non_self_param_ids=non_self_param_ids,
            field_ops=scalar_field_ops,
            field_const_sources=field_const_sources or {},
            field_count=len(scalar_slots),
            output_value_ids=output_value_ids,
            dtype=receiver_scalar_dtype,
            field_dtypes=scalar_slot_dtypes,
        )
        # Field injection rebuilds the function but preserves its sequence
        # metadata, so refresh the table/function correlation after rewriting.
        sequence_helpers, sequence_tables = _sequence_artifacts_from_control(
            control_function
        )
    functions[control_function.name] = control_function
    functions.update(sequence_helpers)
    from .ir_sequence_tables import lower_sequence_aggregate_constants

    lower_sequence_aggregate_constants(functions, sequence_tables)
    # Tensor SSA gets first refusal on shaped subscripts below.  Any scalar
    # Indexed/IndexedStore operations left afterward become universal address
    # primitives; doing that here erased the tensor selection before its
    # repository kernel could see it.
    # Tokenize every string constant to its universal fnv1a token before
    # emission, so a word is a 64-bit value the target expresses like any other
    # constant instead of an inexpressible literal.
    from .ir_string_interning import tokenize_ssa_string_constants

    tokenize_ssa_string_constants(functions, string_table)
    # Section composition and field/index materialization can reconstruct
    # SSAValue wrappers after the control builder's own finish boundary.  Run
    # the same CFG-proven LoopResult reconciliation at the completed section
    # boundary, where every final scalar expression and zero-trip edge exists.
    for function in functions.values():
        rebindings = _canonicalize_non_dominating_loop_result_uses(function)
        if rebindings:
            function.metadata["loop_result_use_rebindings"] = tuple((
                *function.metadata.get("loop_result_use_rebindings", ()),
                *rebindings,
            ))
    record_tables = {}
    reference_slots = {
        int(slot)
        for kind, value_id, slot in field_ops
        if kind == "write"
        and isinstance((field_const_sources or {}).get(int(value_id)), Mapping)
        and (field_const_sources or {})[int(value_id)].get(
            "ssa_reference_identity"
        ) is not None
    }
    if record_identity is not None and field_names:
        from ..transmogrifier.ssa import (
            SSARecordDescriptor,
            SSARecordFieldDescriptor,
            SSARecordFieldStorage,
            SSARecordTable,
        )

        record_fields = []
        nested_record_by_slot = {
            int(slot): (str(identity), int(value_id))
            for slot, identity, value_id in nested_record_fields
        }
        alias_targets = {int(alias): int(target) for alias, target in field_aliases}
        declared_mutability = {
            str(name): bool(mutable)
            for name, mutable in dict(record_field_mutability or {}).items()
        }
        for old_slot, name in enumerate(field_names):
            canonical_slot = alias_targets.get(old_slot, old_slot)
            canonical_name = (
                field_names[canonical_slot]
                if 0 <= canonical_slot < len(field_names)
                else name
            )
            storage_identity = f"{record_identity}.{canonical_name}"
            sequence_id = sequence_field_slots.get(old_slot)
            nested_record = nested_record_by_slot.get(old_slot)
            if nested_record is not None:
                nested_identity, nested_value_id = nested_record
                record_fields.append(SSARecordFieldDescriptor(
                    name,
                    SSARecordFieldStorage.RECORD,
                    storage_identity=storage_identity,
                    # The nested record id is a descriptor correlation, not a
                    # physical SSA value.  Its leaf fields name the actual
                    # caller-owned arenas.
                    value_ids=(),
                    record_id=nested_value_id,
                    dtype=nested_identity,
                ))
            elif sequence_id is not None:
                descriptor = control_sequences[sequence_id]
                value_ids = (
                    *descriptor.column_value_ids,
                    descriptor.length_address_id,
                    descriptor.capacity_value_id,
                    *((descriptor.status_address_id,) if descriptor.status_address_id is not None else ()),
                    *((descriptor.live_flags_value_id,) if descriptor.live_flags_value_id is not None else ()),
                    *(
                        (
                            *descriptor.child_table_pool.column_value_ids,
                            descriptor.child_table_pool.length_value_id,
                            descriptor.child_table_pool.capacity_value_id,
                            descriptor.child_table_pool.row_stride_value_id,
                            *((descriptor.child_table_pool.status_value_id,)
                              if descriptor.child_table_pool.status_value_id is not None else ()),
                            *((descriptor.child_table_pool.live_flags_value_id,)
                              if descriptor.child_table_pool.live_flags_value_id is not None else ()),
                        )
                        if descriptor.child_table_pool is not None else ()
                    ),
                )
                record_fields.append(SSARecordFieldDescriptor(
                    name,
                    SSARecordFieldStorage.SEQUENCE,
                    storage_identity=storage_identity,
                    value_ids=tuple(dict.fromkeys(map(int, value_ids))),
                    sequence_id=sequence_id,
                    writable=descriptor.writable,
                ))
            elif old_slot in compact_slot and self_value_id is not None:
                field_dtype = declared_field_dtypes.get(name)
                record_fields.append(SSARecordFieldDescriptor(
                    name,
                    (
                        SSARecordFieldStorage.REFERENCE
                        if old_slot in reference_slots
                        else SSARecordFieldStorage.SCALAR
                    ),
                    storage_identity=storage_identity,
                    value_ids=(scalar_field_locations[compact_slot[old_slot]][0],),
                    offset=scalar_field_locations[compact_slot[old_slot]][1],
                    dtype=(
                        "opaque_ref"
                        if old_slot in reference_slots else field_dtype
                    ),
                    writable=bool(
                        declared_mutability.get(name, False)
                        and any(
                            kind == "write" and slot == old_slot
                            for kind, _value_id, slot in field_ops
                        )
                    ),
                ))
        if record_fields:
            record_tables[control_function.name] = SSARecordTable({
                int(self_value_id if self_value_id is not None else max(graph_value_ids) + 1):
                    SSARecordDescriptor(
                        int(self_value_id if self_value_id is not None else max(graph_value_ids) + 1),
                        record_identity,
                        tuple(record_fields),
                    )
            })
    from ..transmogrifier.ssa import (
        SSAReferenceDescriptor,
        SSAReferenceKind,
        SSAReferenceTable,
    )

    reference_table = SSAReferenceTable()
    for function in functions.values():
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op != Handler.StaticRef.value:
                    continue
                attributes = instruction.attributes
                reference_table.register(SSAReferenceDescriptor(
                    int(attributes["reference_handle"]),
                    str(attributes["reference_identity"]),
                    SSAReferenceKind(attributes.get(
                        "reference_kind", "static-python"
                    )),
                    bool(attributes.get("host_resident", False)),
                ))
    module = IRModule(
        link_required_ssa_features(functions),
        recursion_table={
            name: dict(function.metadata.get("recursion_table", {}))
            for name, function in functions.items()
            if function.metadata.get("recursion_table")
        },
        sequence_tables=sequence_tables,
        record_tables=record_tables,
        reference_tables=(
            {control_function.name: reference_table}
            if reference_table.references else {}
        ),
    )
    tensor_reference_shortfalls = ()
    if tensor_ssa_reference is not None:
        from .tensor_ssa_lowering import (
            legalize_aggregate_adapters,
            lower_tensor_calls_to_repository_ssa,
            propagate_repository_ssa_call_metadata,
        )

        tensor_reference_shortfalls = tuple(
            SSALoweringShortfall(
                "tensor-ssa-reference",
                item.operation,
                f"{item.function}:{item.block}",
                item.reason,
            )
            for item in lower_tensor_calls_to_repository_ssa(
                module, tensor_ssa_reference
            )
        )
    from .ir_indexing import lower_indexing_to_ssa_addressing

    lower_indexing_to_ssa_addressing(module.functions)
    if tensor_ssa_reference is not None and legalize_aggregate_adapters(module):
        propagate_repository_ssa_call_metadata(module)
    return module, tuple((
        *shortfalls,
        *control_shortfalls,
        *tensor_reference_shortfalls,
    )), section_outputs


def link_verified_source_region_integrals(
    module: IRModule,
    outputs: dict[str, tuple[SSAValue, ...]],
    linked_regions: Mapping[
        tuple[str, ...], tuple[IRModule, Mapping[str, tuple[SSAValue, ...]], Mapping[str, Any]]
    ],
) -> tuple[dict[str, Any], ...]:
    """Install receipt-matched region SSA, retaining current lowering fallback.

    Region token chains are the structural identity.  Integer value IDs are
    checked only as the already-deterministic physical ABI within that named
    compartment; they are never remapped. A stale or unverified candidate is
    ignored and the freshly lowered source region remains authoritative.
    """

    receipts = []

    def value_contract(values: Iterable[SSAValue]) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                int(value.id), tuple(value.shape or ()),
                str(value.dtype or "unknown"), dict(value.accounting or {}),
            )
            for value in values
        )

    scoped_tables = (
        "recursion_table", "deployment_table", "tensor_tables",
        "sequence_tables", "record_tables", "reference_tables", "call_table",
    )
    for function_name, current in tuple(module.functions.items()):
        provenance = dict(current.metadata.get("source_region_integral") or {})
        token_chain = tuple(map(
            str, provenance.get("identity_token_chain") or ()
        ))
        if not token_chain or token_chain not in linked_regions:
            continue
        linked_module, linked_outputs, verification = linked_regions[token_chain]
        record = {
            "ssa_function": str(function_name),
            "identity_token_chain": list(token_chain),
            "status": "fallback",
        }
        if verification.get("status") != "verified":
            record["reason"] = "candidate-is-not-verified"
            receipts.append(record)
            continue
        if tuple(map(
            str, verification.get("identity_token_chain") or ()
        )) != token_chain:
            record["reason"] = "verification-identity-mismatch"
            receipts.append(record)
            continue
        candidate = linked_module.functions.get(str(function_name))
        candidate_outputs = tuple(linked_outputs.get(str(function_name), ()))
        current_outputs = tuple(outputs.get(str(function_name), ()))
        if candidate is None:
            record["reason"] = "candidate-function-is-absent"
            receipts.append(record)
            continue
        if value_contract(candidate.args) != value_contract(current.args):
            record["reason"] = "input-abi-mismatch"
            receipts.append(record)
            continue
        if value_contract(candidate_outputs) != value_contract(current_outputs):
            record["reason"] = "output-abi-mismatch"
            receipts.append(record)
            continue
        candidate_provenance = dict(
            candidate.metadata.get("source_region_integral") or {}
        )
        if tuple(map(
            str, candidate_provenance.get("identity_token_chain") or ()
        )) != token_chain:
            record["reason"] = "candidate-provenance-mismatch"
            receipts.append(record)
            continue
        module.functions[str(function_name)] = candidate
        outputs[str(function_name)] = candidate_outputs
        for table_name in scoped_tables:
            destination = getattr(module, table_name)
            source = getattr(linked_module, table_name)
            if str(function_name) in source:
                destination[str(function_name)] = source[str(function_name)]
            else:
                destination.pop(str(function_name), None)
        record["status"] = "linked"
        record["probe_count"] = int(verification.get("probe_count") or 0)
        receipts.append(record)
    if receipts:
        module.metadata["linked_source_region_integrals"] = tuple(receipts)
    return tuple(receipts)


def lower_precompile_and_control_to_ssa(
    artifact: Any,
    control: ControlProgram,
    *,
    numerical_name: str = "numerical_precompile",
    control_name: str = "planned_control",
    region_programs: dict[int, Any] | None = None,
    hierarchy_plan: PlanClosure | None = None,
    identity_table: Mapping[str, tuple[int, ...]] | None = None,
    function_outputs: tuple[str, ...] = (),
    function_parameters: tuple[str, ...] = (),
    tensor_ssa_reference: Any = None,
) -> PrecompileSSALoweringResult:
    validation = validate_precompile_ssa_compatibility(artifact)
    program = getattr(artifact, "program", artifact)
    numerical, numerical_shortfalls = lower_fused_program_to_ssa(
        program,
        function_name=numerical_name,
    )
    used_ids = {
        value.id
        for value in numerical.args
        for _ in (0,)
    } | {
        instruction.res.id
        for block in numerical.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    # One authority for the authored kernel algorithms: when a tensor code
    # reference is supplied, its (cached) module IS the import -- a second
    # independent import would mint different Function identities and collide
    # at the tensor lowering's linker.
    if tensor_ssa_reference is not None:
        algorithm_functions = dict(tensor_ssa_reference.module.functions)
        algorithm_shortfalls: tuple[Any, ...] = ()
    else:
        algorithm_import = import_llvm_to_repository_ssa(LLVM_SSA_MODULE)
        algorithm_functions = dict(algorithm_import.module.functions)
        algorithm_shortfalls = tuple(algorithm_import.shortfalls)
    functions = dict(algorithm_functions)
    functions[numerical.name] = numerical
    region_callees: dict[int, str] = {}
    region_signatures: dict[
        int, tuple[tuple[int, ...], tuple[int, ...]]
    ] = {}
    region_feed_meta: dict[int, tuple[Meta, ...]] = {}
    region_value_meta: dict[int, Meta] = {}
    region_shortfalls: list[SSALoweringShortfall] = []
    for region_index, region_artifact in sorted(
        (region_programs or {}).items()
    ):
        region_name = f"numerical_region_{int(region_index)}"
        region_program = getattr(
            region_artifact,
            "program",
            region_artifact,
        )
        region_function, shortfalls = lower_fused_program_to_ssa(
            region_program,
            function_name=region_name,
        )
        functions[region_name] = region_function
        region_callees[int(region_index)] = region_name
        region_signatures[int(region_index)] = (
            tuple(sorted(int(value_id) for value_id in region_program.feeds)),
            tuple(
                int(value_id)
                for value_id in region_program.outputs.values()
            ),
        )
        region_feed_meta[int(region_index)] = tuple(
            region_program.meta[int(value_id)]
            for value_id in region_signatures[int(region_index)][0]
        )
        # A region states the dtype and shape of the values it consumes and
        # produces. The control program refers to those same value ids but
        # carries no metadata of its own, so without this the control
        # function's SSA values are shapeless -- which a target that must
        # declare every variable (Fortran) cannot express at all, and which
        # silently degrades an array to a scalar.
        for value_id, meta in (region_program.meta or {}).items():
            region_value_meta.setdefault(int(value_id), meta)
        region_shortfalls.extend(shortfalls)
    if hierarchy_plan is not None:
        for region in hierarchy_plan.items:
            if not (
                isinstance(region, PlanClosure)
                and region.name.startswith("region_")
            ):
                continue
            region_index = int(region.name.split("_", 1)[1])
            if region_index in region_callees:
                continue
            region_name = f"planned_region_{region_index}"
            instructions = list(plan_region_to_ssa_instrs(region))
            produced = {
                int(instruction.res.id)
                for instruction in instructions
                if instruction.res is not None
            }
            consumed = {
                int(argument.id)
                for instruction in instructions
                for argument in instruction.args
            }
            outputs = tuple(sorted(produced - consumed)) or tuple(sorted(produced))
            arguments = [SSAValue(int(value_id)) for value_id in region.captures]
            functions[region_name] = Function(
                region_name,
                arguments,
                {"entry": BasicBlock("entry", instructions)},
            )
            region_callees[region_index] = region_name
            region_signatures[region_index] = (
                tuple(int(value_id) for value_id in region.captures),
                outputs,
            )
            shape_by_id = {
                int(value_id): Meta(tuple(map(int, shape)), str(dtype))
                for value_id, shape, dtype in region.value_shapes
            }
            region_feed_meta[region_index] = tuple(
                shape_by_id.get(int(value_id), Meta((), "unknown"))
                for value_id in region.captures
            )
            known_operations = {handler.value for handler in Handler}
            region_shortfalls.extend(
                SSALoweringShortfall(
                    "planned-region",
                    str(instruction.op),
                    f"{region_name}:entry",
                    "operator has no repository SSA handler",
                )
                for instruction in instructions
                if str(instruction.op) not in known_operations
            )
    if not region_callees:
        region_callees = {
            int(region_index): numerical.name
            for region_index in control.region_indices
        }
    control_function, control_shortfalls = lower_control_program_to_ssa(
        control,
        function_name=control_name,
        first_value_id=max(used_ids, default=-1) + 1,
        region_callees=region_callees,
        region_signatures=region_signatures,
        region_feed_meta=region_feed_meta,
        region_value_meta=region_value_meta,
        output_value_ids=(
            tuple(map(int, program.outputs.values()))
            if region_signatures and not function_outputs else ()
        ),
        named_output_histories={
            str(name): tuple(map(int, (identity_table or {}).get(name, ())))
            for name in function_outputs
        },
        value_name_histories=identity_table,
        parameter_names=function_parameters,
    )
    sequence_helpers, sequence_tables = _sequence_artifacts_from_control(
        control_function
    )
    functions[control_function.name] = control_function
    functions.update(sequence_helpers)
    from .ir_sequence_tables import lower_sequence_aggregate_constants

    lower_sequence_aggregate_constants(functions, sequence_tables)
    # Preserve shaped subscripts until tensor SSA has selected its repository
    # kernels. Scalar structural subscripts are addressed after that pass.
    module = IRModule(
        link_required_ssa_features(functions),
        recursion_table={
            name: dict(function.metadata.get("recursion_table", {}))
            for name, function in functions.items()
            if function.metadata.get("recursion_table")
        },
        sequence_tables=sequence_tables,
    )
    tensor_reference_shortfalls = ()
    if tensor_ssa_reference is not None:
        from .tensor_ssa_lowering import (
            legalize_aggregate_adapters,
            lower_tensor_calls_to_repository_ssa,
            propagate_repository_ssa_call_metadata,
        )

        tensor_reference_shortfalls = tuple(
            SSALoweringShortfall(
                "tensor-ssa-reference",
                item.operation,
                f"{item.function}:{item.block}",
                item.reason,
            )
            for item in lower_tensor_calls_to_repository_ssa(
                module, tensor_ssa_reference
            )
        )
    from .ir_indexing import lower_indexing_to_ssa_addressing

    lower_indexing_to_ssa_addressing(module.functions)
    if tensor_ssa_reference is not None and legalize_aggregate_adapters(module):
        propagate_repository_ssa_call_metadata(module)
    cycles = (
        *find_ssa_cycles(numerical),
        *find_ssa_cycles(control_function),
    )
    return PrecompileSSALoweringResult(
        module,
        validation,
        tuple((
            *numerical_shortfalls,
            *region_shortfalls,
            *control_shortfalls,
            *tensor_reference_shortfalls,
            *(
                SSALoweringShortfall(
                    "llvm",
                    item.opcode,
                    f"{item.function}:{item.block}",
                    item.reason,
                )
                for item in algorithm_shortfalls
            ),
        )),
        tuple(cycles),
    )


def ssa_module_dictionary(module: IRModule) -> dict[str, Any]:
    """Return the repository SSA module as a deterministic plain dictionary."""

    def value(entry: SSAValue | None):
        if entry is None:
            return None
        return {
            "id": int(entry.id),
            "dtype": entry.dtype,
            "shape": tuple(entry.shape),
            "device": entry.device,
            "accounting": dict(entry.accounting),
        }

    return {
        "recursion_table": {
            name: dict(regions)
            for name, regions in module.recursion_table.items()
        },
        "reference_tables": {
            name: [
                table.references[handle].to_mapping()
                for handle in sorted(table.references)
            ]
            for name, table in sorted(module.reference_tables.items())
        },
        "functions": {
            name: {
                "args": [value(argument) for argument in function.args],
                "blocks": {
                    block_name: {
                        "instructions": [
                            {
                                "op": instruction.op,
                                "args": [
                                    value(argument)
                                    for argument in instruction.args
                                ],
                                "result": value(instruction.res),
                                "arg_roles": tuple(
                                    instruction.arg_roles
                                ),
                                "attributes": dict(
                                    instruction.attributes
                                ),
                                "source_span": instruction.source_span,
                            }
                            for instruction in block.instrs
                        ],
                        "successors": tuple(block.successors),
                    }
                    for block_name, block in function.blocks.items()
                },
                "metadata": dict(function.metadata),
            }
            for name, function in module.functions.items()
        }
    }


def merge_repository_ssa_modules(
    primary: IRModule,
    *attachments: IRModule,
) -> IRModule:
    """Compose independently lowered repository SSA surfaces without loss.

    Numerical/control lowering and whole-object lowering are complementary:
    the former owns scheduled regions and imported algorithms, while the
    latter owns class definitions, physical record layouts, method bodies and
    call-frame status.  This merge keeps both in one module and rejects name
    collisions instead of choosing one representation implicitly.
    """

    from ..transmogrifier.ssa import (
        SSAClassTable,
        SSAMachineControlTable,
        SSAMachineIndirectTable,
    )

    modules = (primary, *attachments)

    def merge_named(attribute: str) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for module in modules:
            for name, value in dict(getattr(module, attribute, {}) or {}).items():
                existing = merged.get(str(name))
                if existing is not None and existing != value:
                    raise ValueError(
                        f"conflicting repository SSA {attribute} entry {name!r}"
                    )
                merged[str(name)] = value
        return merged

    classes: dict[str, SSAClassDefinition] = {}
    for module in modules:
        for definition in getattr(module.class_table, "classes", ()):
            existing = classes.get(definition.identity)
            if existing is not None and existing != definition:
                raise ValueError(
                    "conflicting repository SSA class definition "
                    f"{definition.identity!r}"
                )
            classes[definition.identity] = definition

    function_tables = tuple(
        module.function_table for module in modules if len(module.function_table)
    )
    if len({id(table) for table in function_tables}) > 1:
        raise ValueError("cannot merge independently owned SSA function tables")
    function_table = (
        function_tables[0] if function_tables else primary.function_table
    )

    return IRModule(
        merge_named("functions"),
        function_table=function_table,
        class_table=SSAClassTable(tuple(classes.values())),
        recursion_table=merge_named("recursion_table"),
        deployment_table=merge_named("deployment_table"),
        tensor_tables=merge_named("tensor_tables"),
        sequence_tables=merge_named("sequence_tables"),
        record_tables=merge_named("record_tables"),
        reference_tables=merge_named("reference_tables"),
        call_table=merge_named("call_table"),
        machine_control_table=SSAMachineControlTable(tuple(
            link
            for module in modules
            for link in module.machine_control_table.links
        )),
        machine_indirect_table=SSAMachineIndirectTable(tuple(
            link
            for module in modules
            for link in module.machine_indirect_table.links
        )),
    )


__all__ = [
    "PrecompileSSALoweringResult",
    "SSACycle",
    "SSALoweringShortfall",
    "find_ssa_cycles",
    "lower_class_navigation_to_ssa",
    "lower_control_program_to_ssa",
    "lower_control_sections_to_ssa",
    "lower_fused_program_to_ssa",
    "lower_precompile_and_control_to_ssa",
    "merge_repository_ssa_modules",
    "ssa_module_dictionary",
]
