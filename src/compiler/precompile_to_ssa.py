"""Lower an existing Turing numerical precompile and control plan into SSA."""

from __future__ import annotations

import ast
import copy
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import networkx as nx

from .monotonic_ids import GLOBAL_MONOTONIC_IDS
from .id_space import serial_of as _id_serial_of
from .identity_concordance import (
    NEW,
    ConcordanceRefusal,
    Derived,
    IdentityPage,
    Mode,
    Novel,
    Ref,
    Unresolved,
    Unsourced,
    committed_sequence_row_layout,
    concord_sequence_row_dtypes,
    proven_shape_contract_of,
)
from .concordance_declarations import (
    PARAMETER_ABI_DECLARATION,
    PARAMETER_ABI_KIND,
    PARAMETER_ABI_UNDECLARED,
    ParameterAbiKind,
    STORAGE_FORMAL,
    STORAGE_PRODUCED,
    STORAGE_PRODUCER_UNROUTED,
    STORAGE_ROOT,
    STORAGE_ROOT_STAGE,
    STORAGE_ROOT_UNKNOWN,
    StorageRootFact,
    StorageRootKind,
)
from .concordance_declarations import (
    ADDRESS,
    ARM_VERSION_MISSING,
    BINDING_WITHDRAWN,
    CALLSITE_ARGUMENT,
    CANONICAL_VALUE,
    CARRIED_ENTRY_NOT_ATTRIBUTED,
    CARRIED_PORT_VALUE,
    CARRIED_SNAPSHOT,
    CELL_SET,
    CONTROL_CAST,
    CONTROL_CONST,
    CONTROL_EXPRESSION,
    CONTROL_FUNCTION_ROOT,
    CONTROL_PREDICATE,
    CONTROL_SSA,
    CONTROL_SSA_CONDITIONAL,
    CONTROL_SSA_ENTRY,
    CONTROL_SSA_FINISH,
    CONTROL_SSA_LOOP,
    CONTROL_SSA_REGION,
    CONTROL_UNIFORM_DTYPE,
    CONTROL_VALUE_ALIAS,
    CONTROL_VALUE_BINDING,
    CONTROL_VALUE_CONCORDANCE,
    DECLARED_PARAMETER,
    DESCRIPTOR_CELL,
    FIELD_SLOT_ACCESS,
    FUNCTION_OUTPUT,
    FUNCTION_PARAMETER,
    GRAPH_ID_WITHOUT_CANONICAL_CELL,
    INGESTION_VALUE,
    LEXICAL_READ_BINDING,
    LOAD,
    LOOP_CARRIED_BINDING,
    LOOP_CARRIED_ENTRY,
    LOOP_ENTRY_STATE,
    LOOP_RESULT_PORT_BINDING,
    LOOP_RESULT_RECONCILIATION,
    LOOP_RESULT_USE_REBINDING,
    LOOP_RESULT_VERSION,
    REVISION_CAUSE_NOT_ON_BOOK,
    SSA_VALUE_NOT_ON_BOOK,
    NAME_ARM_VERSION_MISSING,
    NAME_BINDING,
    NO_PRODUCER_AT_USE,
    PHI_CONDITIONAL,
    PHI_LOOP_EXIT,
    PHI_LOOP_HEADER,
    PHI_RETURN_MERGE,
    REGION_CALL_RESULT,
    REGION_CAPTURE_BINDING,
    REGION_FEED_CONSUMER,
    REGION_FEED_NO_OPERAND_ROW,
    REGION_FORMAL_SPLIT,
    REGION_SIGNATURE,
    REGION_VALUE_DTYPE,
    RETURN_SLOT_UNRESOLVED_ON_EDGE,
    ROW_COLUMN_PROJECTION,
    SCALAR_ITEM_MERGE,
    SCALAR_PARAMETER,
    SEQUENCE_LENGTH_CELL,
    SSA_FIELD_VERSION,
    SSA_BLOCK,
    SSA_VALUE,
    TABLE_LOOKUP,
    TENSOR_SHAPE_CONCORDANCE,
    VERSIONED_WRITE,
    WHILE_CARRIED_TEST,
    WHILE_TEST_NO_READ_EXPRESSION,
    AliasFact,
    AliasKind,
    BindingKind,
    ControlBinding,
    FieldStateKind,
    ParameterDeclaration,
    RegionSignature,
    SSAValueFact,
    SSAValueOrigin,
)
from .control_source import (
    CallBlock,
    ConditionalBlock,
    ControlBlock,
    ControlExpression,
    ControlProgram,
    ControlSequenceMutation,
    DispatchBlock,
    ResourceScopeBlock,
    ExternalReferenceCallBlock,
    LoopControlBlock,
    LoopBlock,
    ParallelDeployment,
    SequenceBlock,
    SequenceMutationBlock,
    SequenceQueryBlock,
    ScalarFieldWriteBlock,
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
    copy_region_instructions,
    expand_plan_regions,
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


def _case_literal_int(spelling: Any) -> int | None:
    """A tick case label as the integer it denotes (``True``/``False`` are
    the planner's boolean singletons), else None."""

    text = str(spelling).strip()
    if text in {"True", "False"}:
        return int(text == "True")
    try:
        return int(text, 10)
    except ValueError:
        return None


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
        # Watermark from the SERIALS, never the raw numbers.  A raw max over
        # ids that carry group/flag bits (see id_space) lands the watermark
        # in whatever group the largest id belonged to, and every value
        # allocated from it then inherits that group's bits without ever
        # passing through ``compose`` -- a value silently claiming a space
        # it was never minted into.  While ids are unflagged this is exactly
        # the previous behaviour, since a legacy id IS its own serial.
        first_value_id = max(
            (_id_serial_of(value_id) for value_id in authored_ids),
            default=-1,
        ) + 1
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
    nested_tensor: bool = False
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
    nested_tensor_union: dict[int, bool] = {}
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
        nested_tensor_ids = {
            int(sid) for sid in shell.get("nested_tensor_sequence_ids", ())
        }
        for sequence_id, policy, column_count, writable in shell.get(
            "sequence_declarations", ()
        ):
            names = tuple(sequence_names.get(int(sequence_id), ()))
            origin = f"{function_name}{names!r}"
            _observe(sequence_id, column_count, policy, writable, origin)
            sid = int(sequence_id)
            retains_union[sid] = retains_union.get(sid, False) or sid in deletion_ids
            nested_union[sid] = nested_union.get(sid, False) or sid in nested_ids
            nested_tensor_union[sid] = (
                nested_tensor_union.get(sid, False)
                or sid in nested_tensor_ids
            )
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
            nested_tensor_union[sid] = (
                nested_tensor_union.get(sid, False)
                or sid in nested_tensor_ids
            )

    resolved = {
        sid: ResolvedSequenceSchema(
            column_count=column_counts[sid],
            policy=policies[sid],
            writable=writable_union.get(sid, False),
            retains_deleted_rows=retains_union.get(sid, False),
            nested_table=nested_union.get(sid, False),
            nested_tensor=nested_tensor_union.get(sid, False),
        )
        for sid in column_counts
        if sid not in conflicted
    }
    return resolved, tuple(shortfalls)


def _post_derived_or_raw(
    book: Any, page: Any, row: tuple, fact: Any, sources: Any,
    *, stage: Any, mode: Any,
) -> Any:
    """Post ``fact`` DERIVED from ``sources`` when any exist; otherwise write
    it through the raw primitive, which the book tags ``Unsourced`` itself.

    For module-level writers (the region assembly, the reconciliation pass)
    that have no function root to fall back on: a write that can name no
    cell stays visible on the unsourced worklist rather than inventing one.
    """

    cells = tuple(cell for cell in (sources or ()) if isinstance(cell, Ref))
    if cells:
        return book.post(
            page, row, fact, stage=stage, provenance=Derived(cells), mode=mode,
        )
    stored = book.page(page)
    if mode is Mode.CONCORD:
        stored.concord(row, fact)
    else:
        stored.revise(row, fact)
    return book.latest_ref(page, row)


def _post_region_signature(
    book: Any, scope: Any, lexical_read_scope: Any, region_index: int,
    signature: tuple,
) -> Any:
    """``region_signature`` row ``(function scope, region)``: CONCORD
    DERIVED(the ``deployment_region`` cell when the planner posted one, else
    the canonical cells of the region's feeds and outputs -- plan 80 B7's
    fallback, re-derived when the planner's cell lands)."""

    feeds, outputs = signature
    fact = RegionSignature(tuple(map(int, feeds)), tuple(map(int, outputs)))
    row = (scope, int(region_index))
    existing = book.latest_ref(REGION_SIGNATURE, row)
    if existing is not None and (
        book.pages[REGION_SIGNATURE.name].cells.get((row, existing.column))
        == fact
    ):
        return existing
    sources: list = []
    if lexical_read_scope is not None:
        for value_id in (*feeds, *outputs):
            cell = book.latest_ref(
                CANONICAL_VALUE, (tuple(lexical_read_scope), int(value_id)),
            )
            if cell is not None and cell not in sources:
                sources.append(cell)
    return _post_derived_or_raw(
        book, REGION_SIGNATURE, row, fact, tuple(sources),
        stage=CONTROL_SSA_REGION, mode=Mode.CONCORD,
    )


def _function_root_cell(book: Any, scope: Any) -> Ref:
    """The function-root ``cell_set`` row ``(scope, 0)`` of a lowering, posted
    NOVEL(CONTROL_FUNCTION_ROOT) on first need (the same row the builder's
    ``_function_root`` posts, so a module-level minter shares it)."""

    existing = book.latest_ref(CELL_SET, (scope, 0))
    if existing is not None:
        return existing
    return book.post(
        CELL_SET, (scope, 0), (),
        stage=CONTROL_SSA_ENTRY,
        provenance=Novel(CONTROL_FUNCTION_ROOT, ()),
        mode=Mode.CONCORD,
    )


#: Instructions whose result is a view of argument 0's storage (an element,
#: an address into it, a store version of it), not new storage.
_STORAGE_VIEW_OPERATIONS = frozenset({
    "Indexed", "IndexedStore", "GetElementPtr", "Load",
})


def _post_storage_roots(
    book: Any, scope: Any, lexical_read_scope: Any, *,
    formals: Mapping[int, Any], instructions: Iterable[Instr],
    aliases: Mapping[int, int], wanted: Iterable[int],
) -> dict[int, Ref]:
    """Post the ``storage_root`` row of every value in ``wanted`` (and of the
    values its row derives from) and return the cells by value id.

    ``formals``: value id -> the formal's binding cell (or None).  A formal
    is a FORMAL root, an instruction result a PRODUCED root, unless the
    instruction is a view (``_STORAGE_VIEW_OPERATIONS``, or a result that
    declares ``ssa_storage_view``) or the value is a planning alias: those
    are VIEW rows DERIVED from the root cell of the storage they view.
    A value with neither is ``Unresolved(STORAGE_ROOT_UNKNOWN)``.
    """

    producers: dict[int, Instr] = {}
    for instruction in instructions:
        if instruction.res is not None:
            producers.setdefault(int(instruction.res.id), instruction)
    read_scope = (
        None if lexical_read_scope is None else tuple(lexical_read_scope)
    )

    def canonical_cell(value_id: int) -> Ref | None:
        if read_scope is None:
            return None
        return book.latest_ref(CANONICAL_VALUE, (read_scope, int(value_id)))

    cells: dict[int, Ref] = {}
    resolving: set[int] = set()

    def root(value_id: int) -> Ref:
        value_id = int(value_id)
        if value_id in cells:
            return cells[value_id]
        row = (scope, value_id)
        if value_id in resolving:
            # A store version feeding its own base through a carried edge:
            # the cycle names no producer.
            cells[value_id] = book.post(
                STORAGE_ROOT, row, Unresolved(STORAGE_ROOT_UNKNOWN),
                stage=STORAGE_ROOT_STAGE,
                provenance=Unsourced(STORAGE_ROOT_UNKNOWN), mode=Mode.CONCORD,
            )
            return cells[value_id]
        resolving.add(value_id)
        try:
            if value_id in formals:
                source = formals[value_id] or canonical_cell(value_id)
                cell = book.post(
                    STORAGE_ROOT, row,
                    StorageRootFact(StorageRootKind.FORMAL),
                    stage=STORAGE_ROOT_STAGE,
                    provenance=(
                        Novel(STORAGE_FORMAL, (source,)) if source is not None
                        else Unsourced(STORAGE_PRODUCER_UNROUTED)
                    ),
                    mode=Mode.CONCORD,
                )
            elif value_id in aliases and int(aliases[value_id]) != value_id:
                cell = book.post(
                    STORAGE_ROOT, row, StorageRootFact(StorageRootKind.VIEW),
                    stage=STORAGE_ROOT_STAGE,
                    provenance=Derived((root(int(aliases[value_id])),)),
                    mode=Mode.CONCORD,
                )
            elif value_id in producers:
                instruction = producers[value_id]
                viewed = (
                    instruction.args
                    and (
                        str(instruction.op) in _STORAGE_VIEW_OPERATIONS
                        or isinstance(
                            (instruction.res.accounting or {}).get(
                                "ssa_storage_view"
                            ),
                            Mapping,
                        )
                    )
                )
                if viewed:
                    cell = book.post(
                        STORAGE_ROOT, row,
                        StorageRootFact(StorageRootKind.VIEW),
                        stage=STORAGE_ROOT_STAGE,
                        provenance=Derived((root(int(instruction.args[0].id)),)),
                        mode=Mode.CONCORD,
                    )
                else:
                    source = canonical_cell(value_id)
                    cell = book.post(
                        STORAGE_ROOT, row,
                        StorageRootFact(StorageRootKind.PRODUCED),
                        stage=STORAGE_ROOT_STAGE,
                        provenance=(
                            Novel(STORAGE_PRODUCED, (source,))
                            if source is not None
                            else Unsourced(STORAGE_PRODUCER_UNROUTED)
                        ),
                        mode=Mode.CONCORD,
                    )
            else:
                cell = book.post(
                    STORAGE_ROOT, row, Unresolved(STORAGE_ROOT_UNKNOWN),
                    stage=STORAGE_ROOT_STAGE,
                    provenance=Unsourced(STORAGE_ROOT_UNKNOWN),
                    mode=Mode.CONCORD,
                )
        finally:
            resolving.discard(value_id)
        cells[value_id] = cell
        return cell

    for value_id in wanted:
        root(int(value_id))
    return cells


def _storage_formal_root(book: Any, cell: Ref) -> Ref | None:
    """Walk ``cell``'s DERIVED edges on ``storage_root`` to its root; the
    root's cell when it is a FORMAL's storage, else ``None``."""

    page = book.page(STORAGE_ROOT)
    seen: set = set()
    frontier = [cell]
    while frontier:
        ref = frontier.pop()
        if ref.key in seen:
            continue
        seen.add(ref.key)
        fact = page.latest(ref.row)
        if isinstance(fact, StorageRootFact):
            if fact.kind is StorageRootKind.FORMAL:
                return ref
            if fact.kind is StorageRootKind.PRODUCED:
                continue
        frontier.extend(
            source for source, _stage in book.edges_into(ref)
            if source.page.name == STORAGE_ROOT.name
        )
    return None


def post_parameter_abi_kinds(
    book: Any, lexical_read_scope: Any, parameters: Iterable[str], *,
    value_abi: Mapping[str, Any], record_abi: Mapping[str, Any],
) -> dict[str, Ref]:
    """Post each parameter's ``parameter_abi_kind`` row at the signature
    and return the cells by name.

    The declaration is the extraction contract's (``value_abi``: a
    ProgramABI value binding; ``record_abi``: a record binding, the
    receiver included).  A declared kind is NOVEL(PARAMETER_ABI_DECLARATION,
    (version-0 ``name_binding`` cell,)); a parameter nothing declares, or a
    storage spelling outside ``ParameterAbiKind``, is
    ``Unresolved(PARAMETER_ABI_UNDECLARED)`` derived from that cell.
    """

    cells: dict[str, Ref] = {}
    if lexical_read_scope is None:
        return cells
    read_scope = tuple(lexical_read_scope)
    for name in dict.fromkeys(map(str, parameters)):
        binding = book.latest_ref(NAME_BINDING, (lexical_read_scope, name, 0))
        if name in record_abi:
            spelling = "record"
        else:
            spelling = str(dict(value_abi.get(name) or {}).get("storage") or "")
        try:
            kind = ParameterAbiKind(spelling)
        except ValueError:
            kind = None
        row = (read_scope, name)
        if kind is not None and binding is not None:
            fact, provenance = kind, Novel(PARAMETER_ABI_DECLARATION, (binding,))
        elif binding is not None:
            fact = Unresolved(PARAMETER_ABI_UNDECLARED, (binding,))
            provenance = Derived((binding,))
        else:
            fact = kind if kind is not None else Unresolved(PARAMETER_ABI_UNDECLARED)
            provenance = Unsourced(PARAMETER_ABI_UNDECLARED)
        existing = book.latest_ref(PARAMETER_ABI_KIND, row)
        if existing is not None and book.page(PARAMETER_ABI_KIND).latest(row) == fact:
            cells[name] = existing
            continue
        cells[name] = book.post(
            PARAMETER_ABI_KIND, row, fact, stage=STORAGE_ROOT_STAGE,
            provenance=provenance,
            mode=Mode.CONCORD if existing is None else Mode.REVISE,
        )
    return cells


def _mint_ssa_id(
    book: Any, scope: Any, transform: Any, operands: Any, *,
    dtype: Any = None, shape: tuple = (), stage: Any,
) -> int:
    """Mint one SSA id through the book from outside a builder: a NOVEL
    ``ssa_value`` row on ``(scope, NEW)`` whose operand is the one cell in
    ``operands`` (several become a ``cell_set`` row; none is refused, since
    a module-level minter has no function root)."""

    cells = tuple(dict.fromkeys(
        cell for cell in (operands or ()) if isinstance(cell, Ref)
    ))
    if len(cells) == 1:
        operand = cells[0]
    elif cells:
        ordinal = book.page(CELL_SET).scope_row_count(scope) + 1
        operand = book.post(
            CELL_SET, (scope, ordinal), tuple(cell.key for cell in cells),
            stage=stage, provenance=Derived(cells), mode=Mode.CONCORD,
        )
    else:
        raise ConcordanceRefusal(
            f"mint under {scope!r} names no operand cell for {transform.name}"
        )
    ref = book.post(
        SSA_VALUE, (scope, NEW),
        SSAValueFact(dtype, tuple(shape), SSAValueOrigin.MINTED),
        stage=stage, provenance=Novel(transform, (operand,)), mode=Mode.REVISE,
    )
    return int(ref.row[1])


def _post_planned_region_identities(
    book: Any, scope: str, function: Function, signature_cell: Any,
    graph_cell: Any,
) -> None:
    """The identity cells of one planned region function (plan 100, 4.1
    item 2 and 2.6), posted where ``lower_control_sections_to_ssa``
    assembles it.

    ``ssa_value``: every value the region body PRODUCES with no row yet --
    a literal's ``Const`` and every intermediate, whose ids are the graph
    ids they lower -- is adopted as the control builder's ``_value_cell``
    adopts a graph id: ADOPTED_GRAPH_ID DERIVED from the id's
    ``canonical_value`` cell (``graph_cell``, the read-only post-relabel arm
    of ``node_identity_cell``), else
    ``Unsourced(GRAPH_ID_WITHOUT_CANONICAL_CELL)``.  A value that already
    has a row keeps it.  The formals are not posted here: a formal's id is
    the feed id it binds in the caller, under the same control scope, so
    its row IS the feed's (posted by the builder at the call, or the
    feed's ``control_value_binding``) -- one identity, one key.

    ``ssa_block``: the region's one block, DERIVED from the region's
    ``region_signature`` cell."""

    from .concordance_declarations import (
        REGION_VALUE_MINT_UNRECORDED, SSA_BLOCK, SSABlockKind,
    )
    from .id_space import MINTED, has_flag

    for block in function.blocks.values():
        for instruction in block.instrs:
            value = instruction.res
            if value is None or getattr(value, "id", None) is None:
                continue
            row = (scope, int(value.id))
            if book.latest_ref(SSA_VALUE, row) is not None:
                continue
            source = graph_cell(int(value.id))
            # An id with the MINTED flag is no graph id: its minter (the
            # plan's min/max fold) posted no record, and the row says so
            # rather than calling it adopted.
            minted = has_flag(int(value.id), MINTED)
            book.post(
                SSA_VALUE, row,
                SSAValueFact(
                    value.dtype, tuple(value.shape or ()),
                    SSAValueOrigin.MINTED if minted
                    else SSAValueOrigin.ADOPTED_GRAPH_ID,
                ),
                stage=CONTROL_SSA_REGION,
                provenance=(
                    Derived((source,)) if source is not None
                    else Unsourced(REGION_VALUE_MINT_UNRECORDED) if minted
                    else Unsourced(GRAPH_ID_WITHOUT_CANONICAL_CELL)
                ),
                mode=Mode.REVISE,
            )
    if not isinstance(signature_cell, Ref):
        return
    for label in function.blocks:
        book.post(
            SSA_BLOCK, (scope, str(function.name), str(label)),
            SSABlockKind(str(label)), stage=CONTROL_SSA_REGION,
            provenance=Derived((signature_cell,)), mode=Mode.CONCORD,
        )


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
        region_value_ranks: Mapping[int, int] | None = None,
        tensor_shape_concordance_scope: str | None = None,
        lexical_read_scope: Any = None,
        shape_scope: Any = None,
        plan_callsite_bindings: Mapping[
            int, tuple[tuple[int, ...], tuple[int, ...]]
        ] | None = None,
        value_aliases: Mapping[int, int] | None = None,
        inout_value_ids: tuple[int, ...] = (),
        constant_value_ids: tuple[int, ...] = (),
        output_value_ids: tuple[int, ...] = (),
        named_output_histories: Mapping[str, tuple[int, ...]] | None = None,
        value_name_histories: Mapping[str, tuple[int, ...]] | None = None,
        parameter_names: tuple[str, ...] = (),
        sequence_initializations: tuple[tuple[int, str, int], ...] = (),
        sequence_declarations: tuple[tuple[int, str, int, bool], ...] = (),
        sequence_column_dtypes: Mapping[int, tuple[str, ...]] | None = None,
        sequence_record_identities: Mapping[int, str] | None = None,
        sequence_row_record_slots: Mapping[
            int, tuple[tuple[int, str, int], ...]
        ] | None = None,
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
        nested_tensor_sequence_ids: tuple[int, ...] = (),
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
        #: ssa id -> the SSAValue object the book's ``ssa_value`` row names
        #: (plan 80 B2.1): the one private structure the builder keeps,
        #: because emitted instructions hold the objects and mutate them.
        self.ssa_value_objects: dict[int, SSAValue] = {}
        #: The function-root ``cell_set`` cell, posted on first need.
        self._root_cell: Ref | None = None
        #: The stage posts are made under until a construct's lowering
        #: sets its own (conditional, loop, region, finish).
        self.active_stage: Any = CONTROL_SSA_ENTRY
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
        self.region_value_ranks = {
            int(value_id): int(rank)
            for value_id, rank in (region_value_ranks or {}).items()
        }
        if tensor_shape_concordance_scope is None:
            # A lowering with no control scope handed in (the unit-level
            # entry point, a synthetic function): number one on the book so
            # two lowerings of one name never share their rows.
            _label, serial = self._book().mint_scope(f"control:{function_name}")
            tensor_shape_concordance_scope = f"{function_name}@control:{serial}"
        self.tensor_shape_concordance_scope = str(tensor_shape_concordance_scope)
        # The source graph's reduction scope on the identity book: per-read
        # bindings, per-loop carried bindings and per-region capture reads
        # are read from its pages, never passed in as tables.
        self.lexical_read_scope = (
            None if lexical_read_scope is None else tuple(lexical_read_scope)
        )
        #: The shape scope of the graph COPY this function is lowered from
        #: (``identity_concordance.shape_scope_of``): the key every proven
        #: shape this lowering reads is held under.
        self.shape_scope = (
            ("ir_function", str(function_name)) if shape_scope is None
            else tuple(shape_scope)
        )
        #: The loops enclosing the current emission point, as keys of their
        #: ``loop_entry_state`` rows (None for a loop the book cannot key).
        self.enclosing_loop_states: list[Any] = []
        #: Per entered loop key: its carried entries (the lowering's own
        #: SSA values, by entry position) and whether the latch is being
        #: lowered.  Which entry a binding is lives on the book
        #: (``loop_carried_entry``); this is only where the entry's values are.
        self.loop_frames: dict[Any, dict[str, Any]] = {}
        self.value_aliases = {
            int(alias): int(source)
            for alias, source in (value_aliases or {}).items()
        }
        # The live alias map is temporarily rewritten while a loop body is
        # lowered so exit spellings denote the current carried value.  Keep
        # the resolved planning-concordance snapshot immutable for identity
        # and shape decisions that must survive those lexical rewrites.
        self.concorded_value_aliases = dict(self.value_aliases)
        # PLANNING aliases: one ``control_value_alias`` row per resolved
        # planning alias, DERIVED(the ``control_value_concordance`` cell that
        # resolved it, else the canonical cells of alias and source).
        for alias, source in self.value_aliases.items():
            self._post_alias(
                alias, source, AliasKind.PLANNING,
                self._book().latest_ref(
                    CONTROL_VALUE_CONCORDANCE, (str(function_name), int(alias)),
                ),
                stage=CONTROL_SSA_ENTRY,
            )
        self.inout_value_ids = set(map(int, inout_value_ids))
        # Authored literals the control function owns; a use before any
        # region published them is materialized by
        # ``_materialize_control_constants`` from the provisional argument
        # ``external_value`` mints, never invented as an ABI input.
        self.constant_value_ids = set(map(int, constant_value_ids))
        self.output_value_ids = tuple(map(int, output_value_ids))
        self.resource_scopes: list[tuple[ResourceScopeBlock, int]] = []
        # (block, per-slot values) for every source ``return`` lowered below
        # the top level; merged per slot in `function_exit_block` by `finish`.
        self.function_return_edges: list[
            tuple[BasicBlock, tuple[SSAValue | None, ...]]
        ] = []
        self._function_exit: BasicBlock | None = None
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
        self.blocks: dict[str, BasicBlock] = {}
        self.block_counts: dict[str, int] = {}
        #: The ``control_block`` cell (None: the block has no row of its
        #: own) of each control block being lowered, innermost last: the
        #: owner every ``ssa_block`` row ``new_block`` posts derives from.
        self._control_cells: list[Ref | None] = []
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
        self.plan_callsites_by_argument: dict[int, tuple[int, ...]] = {}
        for callsite_id, (argument_ids, _result_ids) in (
            self.plan_callsite_bindings.items()
        ):
            for argument_id in argument_ids:
                value_id = int(argument_id)
                self.plan_callsites_by_argument[value_id] = tuple(dict.fromkeys((
                    *self.plan_callsites_by_argument.get(value_id, ()),
                    int(callsite_id),
                )))
        self.emitted_plan_callsites: set[int] = set()
        # While-latch condition regions may repeat a preheader region whose
        # output list also contains carried seed identities.  The header Phi
        # owns those identities throughout the loop; latch recomputation must
        # project colliding outputs to scratch instead of rebinding the Phi.
        self.preserved_region_output_ids: set[int] = set()
        self.arguments: list[SSAValue] = []
        self.external_values: dict[int, SSAValue] = {}
        #: The SSA value object published under each ``ssa_field_version``
        #: cell this builder posted (keyed by that cell's Ref).  The book row
        #: is the version's identity; this is the object it names, kept the
        #: way ``external_values`` keeps the objects graph ids name.
        self.field_version_values: dict[Any, SSAValue] = {}
        self.scalar_field_effect_destinations: dict[int, tuple[int, str]] = {}

        def index_scalar_field_effects(block: ControlBlock) -> None:
            if isinstance(block, ScalarFieldWriteBlock):
                if block.field_value_id is not None:
                    self.scalar_field_effect_destinations[
                        int(block.effect_node_id)
                    ] = (int(block.field_value_id), str(block.dtype))
                return
            if isinstance(block, SequenceBlock):
                children = block.blocks
            elif isinstance(block, ConditionalBlock):
                children = (block.body, *((block.orelse,) if block.orelse is not None else ()))
            elif isinstance(block, (LoopBlock, WhileBlock)):
                children = (block.body, *block.terminal_controls)
            elif isinstance(block, CallBlock):
                children = (block.callee,)
            elif isinstance(block, ResourceScopeBlock):
                children = (block.body, *block.cleanup)
            elif isinstance(block, ParallelDeployment):
                children = tuple(lane.body for lane in block.lanes)
            elif isinstance(block, StateMachineTick):
                children = tuple(body for _case, body in block.cases)
                if block.default is not None:
                    children = (*children, block.default)
            else:
                children = ()
            for child in children:
                index_scalar_field_effects(child)

        index_scalar_field_effects(program.root)
        _alias_debug = os.environ.get("TURING_DEBUG_ALIAS_BINDING")
        if _alias_debug and _alias_debug in str(function_name):
            import traceback as _traceback

            class _TracedExternals(dict):
                def __setitem__(_self, key, value):
                    if value is not None and int(value.id) != int(key):
                        frames = _traceback.extract_stack(limit=6)[:-1]
                        print(
                            f"DEBUG-ALIAS-BINDING {function_name}: "
                            f"external_values[{key}] = value {int(value.id)} "
                            f"accounting={dict(value.accounting or {})} at "
                            + " <- ".join(f"{f.name}:{f.lineno}" for f in reversed(frames)),
                            file=sys.stderr, flush=True,
                        )
                    super().__setitem__(key, value)

            self.external_values = _TracedExternals()
        self.declared_parameter_only_ids: set[int] = set()
        self.validation_contracts: list[dict[str, object]] = []
        self.control_identity_receipts: list[tuple[int, int, str]] = []
        self.table_lookup_ownership_receipts: list[
            tuple[int, str, tuple[int, ...]]
        ] = []
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
        # Record-typed columns of fixed-width rows: (authored column,
        # identity, physical width) per resident sequence.
        self.sequence_row_record_slots = {
            int(sequence_id): tuple(
                (int(position), str(identity), int(width))
                for position, identity, width in slots
            )
            for sequence_id, slots in (
                sequence_row_record_slots or {}
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
        # A source value may seed several logical loop bindings.  Conditional
        # alias publication must not overwrite one binding's update merely
        # because another binding assigned that value as its RHS.
        self.protected_loop_alias_sources: list[frozenset[int]] = []
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
            # UNIFORM: the control program's declared dtype for the uniform
            # (``control_uniform_dtype``, posted at entry) is the cause.
            self._bind(
                int(uniform.value_id), value, BindingKind.UNIFORM,
                self._declared_cell(
                    CONTROL_UNIFORM_DTYPE.name,
                    (str(self.tensor_shape_concordance_scope), int(uniform.value_id)),
                ),
            )
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
            # PARAMETER_SEED: the parameter's version-0 ``name_binding`` cell
            # is the cause; the seed is declared CALL_ONLY until a use.
            binding_cell = (
                None if self.lexical_read_scope is None
                else self._book().latest_ref(
                    NAME_BINDING, (self.lexical_read_scope, str(name), 0),
                )
            )
            self._bind(value_id, value, BindingKind.PARAMETER_SEED, binding_cell)
            self.arguments.append(value)
            self.declared_parameter_only_ids.add(value_id)
            self._declare_parameter(
                value_id, ParameterDeclaration.CALL_ONLY, binding_cell,
            )
        signature_ids = {
            value_id
            for feeds, outputs in self.region_signatures.values()
            for value_id in (*feeds, *outputs)
        }
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
                dtype=str(self._value_from_meta(value_id).dtype or "unknown"),
                transform=ROW_COLUMN_PROJECTION, operands=(int(value_id),),
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
        nested_tensor_sequence_ids = set(map(
            int, nested_tensor_sequence_ids
        ))
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
        from .identity_concordance import commit_sequence_contract

        def authored_sequence_mutations(
            block: ControlBlock,
        ) -> tuple[ControlSequenceMutation, ...]:
            """Return the exact resident effects carried by this control tree.

            Loop mutations and lexical mutation blocks are two placements of
            the same authored effect in some plans.  The ProcessGraph effect
            identity is therefore the deduplication key; encounter order is
            otherwise retained so conflicting contracts remain deterministic.
            """

            found: list[ControlSequenceMutation] = []
            seen_effects: set[int] = set()

            def visit(candidate: ControlBlock | None) -> None:
                if candidate is None:
                    return
                mutations: tuple[ControlSequenceMutation, ...] = ()
                if isinstance(candidate, SequenceMutationBlock):
                    mutations = (candidate.mutation,)
                elif isinstance(candidate, (LoopBlock, WhileBlock)):
                    mutations = tuple(candidate.sequence_mutations)
                for mutation in mutations:
                    effect_id = int(mutation.effect_node_id)
                    if effect_id not in seen_effects:
                        seen_effects.add(effect_id)
                        found.append(mutation)
                if isinstance(candidate, SequenceBlock):
                    for child in candidate.blocks:
                        visit(child)
                if isinstance(candidate, WhileBlock):
                    visit(candidate.condition)
                if isinstance(candidate, (ConditionalBlock, LoopBlock, WhileBlock)):
                    visit(candidate.body)
                if isinstance(candidate, ConditionalBlock):
                    visit(candidate.orelse)
                if isinstance(candidate, CallBlock):
                    visit(candidate.callee)

            visit(block)
            return tuple(found)

        control_mutations = authored_sequence_mutations(program.root)

        static_sequence_capacities: dict[int, int] = {}

        def collect_static_sequence_capacities(
            block: ControlBlock | None,
        ) -> None:
            if block is None:
                return
            if isinstance(block, LoopBlock):
                try:
                    extent = len(range(
                        int(block.start), int(block.stop), int(block.step),
                    ))
                except (TypeError, ValueError, ZeroDivisionError):
                    extent = None
                if extent is not None:
                    for mutation in block.sequence_mutations:
                        sequence_id = int(mutation.sequence_value_id)
                        incumbent = static_sequence_capacities.get(sequence_id)
                        if incumbent is not None and incumbent != extent:
                            raise ValueError(
                                "one resident sequence has conflicting static "
                                f"producer extents: sequence={sequence_id} "
                                f"{incumbent} versus {extent}"
                            )
                        static_sequence_capacities[sequence_id] = extent
            if isinstance(block, SequenceBlock):
                for child in block.blocks:
                    collect_static_sequence_capacities(child)
            if isinstance(block, WhileBlock):
                collect_static_sequence_capacities(block.condition)
            if isinstance(block, (ConditionalBlock, LoopBlock, WhileBlock)):
                collect_static_sequence_capacities(block.body)
            if isinstance(block, ConditionalBlock):
                collect_static_sequence_capacities(block.orelse)
            if isinstance(block, CallBlock):
                collect_static_sequence_capacities(block.callee)

        collect_static_sequence_capacities(program.root)
        self.static_sequence_capacities = static_sequence_capacities

        for sequence_id, policy, column_count, writable in sequence_declarations:
            committed = commit_sequence_contract(
                function_name,
                int(sequence_id),
                str(policy),
                int(column_count),
                bool(writable),
                source="SSA sequence declaration",
            )
            authored_column_dtypes = tuple(
                (sequence_column_dtypes or {}).get(int(sequence_id), ())
            )
            column_dtypes: list[str | None] = [
                authored_column_dtypes[column]
                if column < len(authored_column_dtypes) else None
                for column in range(int(column_count))
            ]
            column_shapes: list[tuple[int, ...]] = [
                () for _column in range(int(column_count))
            ]
            row_layout = committed_sequence_row_layout(
                self.shape_scope, int(sequence_id),
            )
            if row_layout is not None:
                if len(row_layout.column_shapes) != int(column_count):
                    raise ValueError(
                        "sequence row layout width disagrees with its "
                        f"physical contract in {function_name}: sequence "
                        f"{int(sequence_id)} has "
                        f"{len(row_layout.column_shapes)} layout column(s) "
                        f"and {int(column_count)} physical column(s)"
                    )
                column_shapes[:] = row_layout.column_shapes
                for column, dtype in enumerate(row_layout.column_dtypes):
                    if column_dtypes[column] in {None, "unknown"}:
                        column_dtypes[column] = dtype
            # A retained comprehension appends the numerical value produced
            # by its body.  That exact mutation edge owns the shape of one
            # resident row; the sequence handle is not the row and must not
            # inherit the row shape as its complete extent.  The shape proof
            # is an identity-book fact produced by the source transformation
            # graph.  Do not reconstruct it from ``region_value_meta``: that
            # id-keyed summary is a stage-local second key and was precisely
            # how a proven ``(3,)`` reduction result became a scalar row here.
            for mutation in control_mutations:
                if (
                    int(mutation.sequence_value_id) != int(sequence_id)
                    or str(mutation.operator) not in {"append", "add"}
                    or len(tuple(mutation.argument_value_ids))
                    != int(column_count)
                ):
                    continue
                for column, argument_id in enumerate(
                    mutation.argument_value_ids
                ):
                    row_contract = proven_shape_contract_of(
                        self.shape_scope, int(argument_id),
                    )
                    if row_contract is None:
                        continue
                    proposed_shape, proposed_dtype = row_contract
                    incumbent_shape = column_shapes[column]
                    if incumbent_shape and incumbent_shape != proposed_shape:
                        self.shortfalls.append(SSALoweringShortfall(
                            "ssa-sequence", "conflicting-row-shape",
                            f"{function_name}.sequence_declaration",
                            f"sequence value {int(sequence_id)} column "
                            f"{column} receives both {incumbent_shape!r} "
                            f"and {proposed_shape!r}",
                        ))
                        continue
                    column_shapes[column] = proposed_shape
                    if column_dtypes[column] in {None, "unknown"}:
                        column_dtypes[column] = str(proposed_dtype)
            for _result_id, query_id, lookup_sequence_id in table_lookups:
                if int(lookup_sequence_id) != int(sequence_id):
                    continue
                query_ids = query_id if isinstance(query_id, tuple) else (query_id,)
                for column, query_value_id in enumerate(query_ids):
                    meta = self.region_value_meta.get(int(query_value_id))
                    if meta is not None and column < len(column_dtypes) and column_dtypes[column] in {None, "unknown"}:
                        column_dtypes[column] = str(meta.dtype)
            for _effect_id, key_id, value_id, store_sequence_id in table_stores:
                if int(store_sequence_id) != int(sequence_id):
                    continue
                key_ids = key_id if isinstance(key_id, tuple) else (key_id,)
                for column, key_value_id in enumerate(key_ids):
                    meta = self.region_value_meta.get(int(key_value_id))
                    if meta is not None and column < len(column_dtypes) and column_dtypes[column] in {None, "unknown"}:
                        column_dtypes[column] = str(meta.dtype)
                value_meta = self.region_value_meta.get(int(value_id))
                if value_meta is not None and column_dtypes and column_dtypes[-1] in {None, "unknown"}:
                    column_dtypes[-1] = str(value_meta.dtype)
            self._sequence_descriptor(
                int(sequence_id),
                policy=committed.policy,
                writable=committed.writable,
                location=f"{function_name}.sequence_declaration",
                column_count=committed.column_count,
                retains_deleted_rows=int(sequence_id) in deletion_sequence_ids,
                nested_table=int(sequence_id) in nested_sequence_ids,
                nested_tensor=(
                    int(sequence_id) in nested_tensor_sequence_ids
                ),
                nested_value_dtype=nested_value_dtypes.get(int(sequence_id)),
                element_dtype=(
                    "int" if int(sequence_id) in self.joined_sequence_ids
                    else None
                ),
                column_dtypes=tuple(column_dtypes),
                column_shapes=tuple(column_shapes),
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
            length_cell = self.sequence_storage_values[int(sequence_id)][
                len(descriptor.column_value_ids)
            ]
            zero_index = self.constant_value(0, length_cell)
            zero_length = self.constant_value(0, length_cell)
            length_address = self.fresh_value(
                dtype="ptr", transform=ADDRESS, operands=(length_cell,),
            )
            self.emit(
                Handler.GetElementPtr,
                [length_cell, zero_index],
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
        for sequence_id in sorted(self.joined_sequence_ids):
            if int(sequence_id) not in self.sequence_descriptors:
                continue
            # The flat view's sequence id is a compiler-minted identity: mint
            # it through the book from the joined sequence's own cell.
            flat_sequence_id = int(self.fresh_value(
                dtype="int", transform=DESCRIPTOR_CELL,
                operands=(int(sequence_id), self._sequence_contract_cell(int(sequence_id))),
            ).id)
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
            flat_length_address = self.fresh_value(
                dtype="ptr", transform=ADDRESS, operands=(flat_length,),
            )
            self.emit(
                Handler.GetElementPtr,
                [flat_length, self.constant_value(0, flat_length)],
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
                nested_tensor=(
                    int(sequence_id) in nested_tensor_sequence_ids
                ),
                nested_value_dtype=nested_value_dtypes.get(int(sequence_id)),
                element_dtype=element_dtype,
            )
            if descriptor is None:
                continue
            length_cell = self.sequence_storage_values[int(sequence_id)][
                len(descriptor.column_value_ids)
            ]
            zero = self.constant_value(0, length_cell)
            zero_index = self.constant_value(0, length_cell)
            length_address = self.fresh_value(
                dtype="ptr", transform=ADDRESS, operands=(length_cell,),
            )
            self.emit(
                Handler.GetElementPtr,
                [length_cell, zero_index],
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
                    index = self.constant_value(int(offset), length_cell)
                    for column, literal in enumerate(tuple(row)):
                        column_cell = self.sequence_storage_values[
                            int(sequence_id)
                        ][column]
                        address = self.fresh_value(
                            dtype="ptr", transform=ADDRESS,
                            operands=(column_cell, index),
                        )
                        self.emit(
                            Handler.GetElementPtr,
                            [column_cell, index],
                            address,
                            attributes={
                                "binding": "ssa_sequence_literal_table"
                            },
                        )
                        value = self.fresh_value(
                            dtype=descriptor.column_dtypes[column],
                            transform=CONTROL_CONST, operands=(address,),
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
                    [self.constant_value(len(rows), length_cell), length_address],
                    attributes={
                        "binding": "ssa_sequence_literal_table_length"
                    },
                )
            elif str(policy).startswith("literal_bytes="):
                payload = bytes.fromhex(str(policy).split("=", 1)[1])
                byte_column = self.sequence_storage_values[int(sequence_id)][0]
                for offset, byte in enumerate(payload):
                    index = self.constant_value(int(offset), byte_column)
                    address = self.fresh_value(
                        dtype="ptr", transform=ADDRESS,
                        operands=(byte_column, index),
                    )
                    self.emit(
                        Handler.GetElementPtr,
                        [byte_column, index],
                        address,
                        attributes={"binding": "ssa_sequence_literal_bytes"},
                    )
                    value = self.fresh_value(
                        dtype="int", transform=CONTROL_CONST, operands=(address,),
                    )
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

            helper_name = self._sequence_helper_name(
                int(sequence_id), "contains",
            )
            lowering = lower_sequence_contains(
                descriptor,
                function_name=helper_name,
                first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
            )
            self._register_sequence_lowering(lowering)
            query = self.external_value(int(query_id))
            contract_cell = self._sequence_contract_cell(int(sequence_id))
            call_result = (
                self.fresh_value(
                    dtype="bool", transform=CONTROL_PREDICATE,
                    operands=(query, contract_cell),
                )
                if negate else SSAValue(int(result_id), dtype="bool")
            )
            if not negate:
                self._bind(
                    int(result_id), call_result, BindingKind.SEQUENCE_RESULT,
                    query, contract_cell,
                )
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
                self._bind(
                    int(result_id), result, BindingKind.SEQUENCE_RESULT,
                    call_result,
                )
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
            # The owner is the function whose regions or explicitly bound
            # source calls consume the result.  A lookup used only as a call
            # argument has no numerical-region metadata, but the plan binding
            # is equally exact local provenance.
            callsite_owners = self.plan_callsites_by_argument.get(
                int(result_id), ()
            )
            if (
                int(result_id) not in self.region_value_meta
                and not callsite_owners
            ):
                continue
            ownership = "region"
            owner_ids: tuple[int, ...] = ()
            if callsite_owners:
                ownership = "source_call"
                owner_ids = tuple(map(int, callsite_owners))
            self.table_lookup_ownership_receipts.append((
                int(result_id), ownership, owner_ids,
            ))
            self._emit_table_lookup(
                result_id, query_id, sequence_id,
                ownership=ownership,
                owner_ids=owner_ids,
            )
        for effect_id, key_id, value_id, sequence_id in table_stores:
            if ("store", (effect_id, key_id, value_id, sequence_id)) in scheduled_table_operations:
                continue
            self._emit_table_store(effect_id, key_id, value_id, sequence_id)
        for effect_id, key_id, sequence_id, storage_identity in table_deletions:
            if ("delete", (effect_id, key_id, sequence_id, storage_identity)) in scheduled_table_operations:
                continue
            self._emit_table_delete(effect_id, key_id, sequence_id, storage_identity)
        self.active_stage = CONTROL_SSA

    def _sequence_helper_name(
        self, sequence_id: int, operation: str, *related_ids: int,
    ) -> str:
        """Name function-local sequence storage in the repository namespace.

        Sequence identities are deliberately local to one lowered function.
        Their helper symbols therefore have to retain that owner identity too;
        spelling only the numeric id lets an unrelated function's helper
        overwrite this one when the per-function modules are assembled.
        """

        suffix = "".join(f"_{int(value_id)}" for value_id in related_ids)
        return (
            f"{self.function_name}__ssa_sequence_{int(sequence_id)}_"
            f"{operation}{suffix}"
        )

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
        self, result_id: int, query_id: int | tuple[int, ...], sequence_id: int,
        *, ownership: str = "scheduled", owner_ids: tuple[int, ...] = (),
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
        helper_name = self._sequence_helper_name(
            int(sequence_id),
            "lookup_or_default" if default_literal is not None else "lookup",
        )
        lowering = lower_table_lookup(
            descriptor,
            function_name=helper_name,
            default_parameter=default_literal is not None,
            first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
        )
        self._register_sequence_lowering(lowering)
        pool = descriptor.child_table_pool
        tensor_child = bool(
            pool is not None
            and not pool.key_columns
            and len(pool.column_value_ids) == 1
        )
        contract_cell = self._sequence_contract_cell(int(sequence_id))
        query_values = self._table_query_values(query_id)
        result = (
            self.fresh_value(
                dtype="int", transform=TABLE_LOOKUP,
                operands=(contract_cell, *query_values),
            )
            if tensor_child
            else SSAValue(
                int(result_id),
                dtype=("int" if pool is not None
                       else descriptor.column_dtypes[
                           next(
                               column
                               for column in range(
                                   len(descriptor.column_value_ids)
                               )
                               if column not in descriptor.key_columns
                           )
                       ]),
            )
        )
        if not tensor_child:
            self._bind(
                int(result_id), result, BindingKind.SEQUENCE_RESULT,
                contract_cell, *query_values,
            )
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
            default_value = self.fresh_value(
                dtype=str(default_dtype), transform=CONTROL_CONST,
                operands=(contract_cell, *query_values),
            )
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
                *query_values,
                *default_operands,
            ],
            result,
            attributes={
                "callee": helper_name,
                "source_linked": True,
                "ssa_sequence_operation": "lookup",
                "sequence_id": int(sequence_id),
                "ssa_lookup_ownership": str(ownership),
                "ssa_lookup_owner_ids": tuple(map(int, owner_ids)),
            },
        )
        if pool is not None:
            self.child_table_selections[int(result_id)] = (
                pool, result
            )
        if tensor_child:
            # A keyed Tensor value is a child-row handle at the outer lookup
            # boundary and a span at every numerical consumer. Materialize
            # that existing child-pool address contract here, while the exact
            # handle and pool identities are both present. No backend should
            # reconstruct a Python mapping or guess which arena the handle
            # selects.
            row_stride = self.external_value(pool.row_stride_value_id)
            offset = self.fresh_value(
                dtype="int64", transform=CONTROL_EXPRESSION,
                operands=(result, row_stride),
            )
            row_base = self.produced_value(
                int(result_id),
                dtype=str(pool.column_dtypes[0] or "unknown"),
                claim_provisional_definition=True,
                cause=(result, offset),
            )
            self.emit(
                Handler.Mul,
                [result, row_stride],
                offset,
                attributes={"binding": "keyed_tensor_row_offset"},
            )
            self.emit(
                Handler.GetElementPtr,
                [self.external_value(pool.column_value_ids[0]), offset],
                row_base,
                attributes={"binding": "keyed_tensor_row_base"},
            )
            if (
                pool.shape_value_id is None
                or pool.rank_value_id is None
                or pool.shape_stride_value_id is None
            ):
                raise ValueError(
                    f"keyed Tensor sequence {sequence_id} has no concorded "
                    "shape/rank child-pool contract"
                )
            shape_stride = self.external_value(pool.shape_stride_value_id)
            shape_offset = self.fresh_value(
                dtype="int64", transform=CONTROL_EXPRESSION,
                operands=(result, shape_stride),
            )
            shape_address = self.fresh_value(
                dtype="int32", transform=ADDRESS, operands=(shape_offset,),
            )
            rank_address = self.fresh_value(
                dtype="int32", transform=ADDRESS, operands=(result,),
            )
            rank = self.fresh_value(
                dtype="int32", transform=LOAD, operands=(rank_address,),
            )
            element_count = self.fresh_value(
                dtype="int32", transform=LOAD, operands=(result,),
            )
            self.emit(
                Handler.Mul,
                [result, shape_stride],
                shape_offset,
                attributes={"binding": "keyed_tensor_shape_offset"},
            )
            self.emit(
                Handler.GetElementPtr,
                [self.external_value(pool.shape_value_id), shape_offset],
                shape_address,
                attributes={"binding": "keyed_tensor_row_shape"},
            )
            self.emit(
                Handler.GetElementPtr,
                [self.external_value(pool.rank_value_id), result],
                rank_address,
                attributes={"binding": "keyed_tensor_row_rank_address"},
            )
            self.emit(
                Handler.Load,
                [rank_address],
                rank,
                attributes={"binding": "keyed_tensor_row_rank"},
            )
            length_address = self.fresh_value(
                dtype="int32", transform=ADDRESS, operands=(result,),
            )
            self.emit(
                Handler.GetElementPtr,
                [self.external_value(pool.length_value_id), result],
                length_address,
                attributes={"binding": "keyed_tensor_row_length_address"},
            )
            self.emit(
                Handler.Load,
                [length_address],
                element_count,
                attributes={"binding": "keyed_tensor_row_length"},
            )
            row_base.accounting = {
                **dict(row_base.accounting or {}),
                "program_abi_storage": "span",
                "program_abi_rank": max(
                    1, int(self.region_value_ranks.get(int(result_id), 1))
                ),
                "tensor_metadata_state": "dynamic",
                "tensor_shape_value_id": int(shape_address.id),
                "tensor_rank_value_id": int(rank.id),
                "tensor_element_count_value_id": int(element_count.id),
                "keyed_tensor_handle_value_id": int(result.id),
            }
            # This lookup is the stage that owns the physical child-span
            # identities.  Commit the complete contract under the authored
            # result identity so the planned numerical occurrence can recover
            # it without rediscovering shape from flat storage.
            # DERIVED(the lookup's ``ssa_value`` cell and the cells it names).
            self._post_tensor_shape_contract(
                int(result_id),
                {
                    "program_abi_storage": "span",
                    "program_abi_rank": int(
                        row_base.accounting["program_abi_rank"]
                    ),
                    "tensor_metadata_state": "dynamic",
                    "tensor_shape_value_id": int(shape_address.id),
                    "tensor_rank_value_id": int(rank.id),
                    "tensor_element_count_value_id": int(element_count.id),
                    "source": "control-keyed-tensor-lookup",
                },
                result, shape_address, rank, element_count,
            )
            self._bind(
                int(result_id), row_base, BindingKind.SEQUENCE_RESULT,
                result, offset,
            )

    def _post_tensor_shape_contract(
        self, value_id: int, contract: dict, *cells: Any,
    ) -> Ref:
        """``tensor_shape_concordance`` row ``(control scope, value)``: a
        REVISE DERIVED from ``cells`` (and the incumbent, when the new
        contract is merged over one), replacing the silent page-wide
        column write."""

        book = self._book()
        row = (str(self.tensor_shape_concordance_scope), int(value_id))
        incumbent = book.latest_ref(TENSOR_SHAPE_CONCORDANCE, row)
        sources = self._cells(incumbent, *cells, self._canonical_cell(value_id))
        if not sources:
            sources = (self._function_root(),)
        return book.post(
            TENSOR_SHAPE_CONCORDANCE, row, dict(contract),
            stage=self.active_stage,
            provenance=Derived(sources),
            mode=Mode.REVISE,
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

        helper_name = self._sequence_helper_name(int(sequence_id), "store")
        lowering = lower_table_store(
            descriptor,
            function_name=helper_name,
            first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
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
        length = self.fresh_value(
            dtype="int64", transform=LOAD, operands=(length_cell,),
        )
        self.emit(
            Handler.Load, [length_cell], length,
            attributes={
                "binding": "ssa_sequence_row_length",
                "sequence_id": int(sequence_id),
            },
        )
        normalized = self.fresh_value(
            dtype="int64", transform=CONTROL_EXPRESSION, operands=(length,),
        )
        self.emit(
            Handler.Add, [length, self.constant_value(int(literal), length)], normalized,
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
            column_value = self.external_value(int(column_id))
            address = self.fresh_value(
                dtype="ptr", transform=ADDRESS, operands=(column_value, index),
            )
            self.emit(
                Handler.GetElementPtr,
                [column_value, index],
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
        self._bind(
            int(result_id), self.external_value(int(sequence_id)),
            BindingKind.SEQUENCE_RESULT,
            self._sequence_contract_cell(int(sequence_id)), index,
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
                    f"{self.function_name}__ssa_child_table_"
                    f"{int(match.group(1))}_delete"
                )
                lowering = lower_child_table_delete(
                    pool,
                    function_name=helper_name,
                    first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
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
            helper_name = self._sequence_helper_name(
                int(sequence_id), "delete_first",
            )
            lowering = lower_table_delete_first(
                descriptor,
                function_name=helper_name,
                first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
            )
        else:
            helper_name = self._sequence_helper_name(
                int(sequence_id), "delete",
            )
            lowering = lower_table_delete(
                descriptor,
                function_name=helper_name,
                first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
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

        helper_name = self._sequence_helper_name(int(sequence_id), "fill")
        lowering = lower_sequence_fill(
            descriptor,
            function_name=helper_name,
            first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
        )
        self._register_sequence_lowering(lowering)
        contract_cell = self._sequence_contract_cell(int(sequence_id))
        value = self.fresh_value(
            dtype=str(descriptor.column_dtypes[0]), transform=CONTROL_CONST,
            operands=(contract_cell,),
        )
        self.emit(Handler.Const, [], value, attributes={"value": literal})
        status = self.fresh_value(
            dtype="int", transform=REGION_CALL_RESULT,
            operands=(contract_cell, value),
        )
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

        helper_name = self._sequence_helper_name(
            int(sequence_id), "append_fill",
        )
        lowering = lower_sequence_append_fill(
            descriptor,
            function_name=helper_name,
            first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
        )
        self._register_sequence_lowering(lowering)
        contract_cell = self._sequence_contract_cell(int(sequence_id))
        value = self.fresh_value(
            dtype=str(descriptor.column_dtypes[0]), transform=CONTROL_CONST,
            operands=(contract_cell,),
        )
        self.emit(Handler.Const, [], value, attributes={"value": literal})
        status = self.fresh_value(
            dtype="int", transform=REGION_CALL_RESULT,
            operands=(contract_cell, value),
        )
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

        helper_name = self._sequence_helper_name(
            int(destination_id), "append_slice", int(source_id),
        )
        lowering = lower_sequence_append_slice(
            destination,
            source,
            function_name=helper_name,
            first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
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
        source_cell = self._sequence_contract_cell(int(source_id))
        if lower_id is None:
            lower_value = self.fresh_value(
                dtype="int", transform=CONTROL_CONST, operands=(source_cell,),
            )
            self.emit(
                Handler.Const, [], lower_value, attributes={"value": 0}
            )
        else:
            lower_value = self.external_value(int(lower_id), dtype="int")
        if upper_id is None:
            upper_value = self.fresh_value(
                dtype="int", transform=CONTROL_CONST, operands=(source_cell,),
            )
            self.emit(
                Handler.Const, [], upper_value,
                attributes={"value": 2 ** 31 - 1},
            )
        else:
            upper_value = self.external_value(int(upper_id), dtype="int")
        status = self.fresh_value(
            dtype="int", transform=REGION_CALL_RESULT,
            operands=(*storage, lower_value, upper_value),
        )
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
        length_address = storage[len(descriptor.column_value_ids)]
        zero = self.fresh_value(
            dtype="int64", transform=CONTROL_CONST, operands=(length_address,),
        )
        self.emit(Handler.Const, [], zero, attributes={"value": 0})
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

        helper_name = self._sequence_helper_name(
            int(destination_id), "pack_bits", int(source_id),
        )
        lowering = lower_sequence_pack_bits(
            destination,
            source,
            function_name=helper_name,
            first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
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
        status = self.fresh_value(
            dtype="int", transform=REGION_CALL_RESULT, operands=tuple(storage),
        )
        if width_literal:
            width = self.fresh_value(
                dtype="int", transform=CONTROL_CONST, operands=tuple(storage),
            )
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

        helper_name = self._sequence_helper_name(int(sequence_id), "prepend")
        lowering = lower_sequence_prepend(
            descriptor,
            function_name=helper_name,
            first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
        )
        self._register_sequence_lowering(lowering)
        status = self.fresh_value(
            dtype="int", transform=REGION_CALL_RESULT,
            operands=(self._sequence_contract_cell(int(sequence_id)),),
        )
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

        helper_name = self._sequence_helper_name(
            int(destination_id), "prepend_packed", int(source_id),
        )
        lowering = lower_sequence_prepend_packed_bytes(
            destination,
            source,
            function_name=helper_name,
            first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
        )
        self._register_sequence_lowering(lowering)
        storage = tuple({
            value.id: value
            for value in (
                *self.sequence_storage_values[int(destination_id)],
                *self.sequence_storage_values[int(source_id)],
            )
        }.values())
        width_value = self.fresh_value(
            dtype="int", transform=CONTROL_CONST, operands=tuple(storage),
        )
        self.emit(
            Handler.Const, [], width_value,
            attributes={"value": int(byte_width)},
        )
        status = self.fresh_value(
            dtype="int", transform=REGION_CALL_RESULT,
            operands=(*storage, width_value),
        )
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

    def external_value(
        self,
        value_id: int,
        *,
        dtype: str | None = None,
        follow_aliases: bool = True,
    ) -> SSAValue:
        value_id = int(value_id)
        field_effect = self.scalar_field_effect_destinations.get(value_id)
        if field_effect is not None and value_id not in self.external_values:
            field_value_id, field_dtype = field_effect
            if field_value_id != value_id:
                # Before its lexical Store, a SetAttr version denotes the
                # incumbent value in the same resident field.  This matters
                # when an in-place update uses one graph identity for both the
                # pre-branch and written versions.
                incumbent = self.external_value(field_value_id, dtype=field_dtype)
                # FIELD_INCUMBENT: the incumbent's own binding cell is the
                # cause (its ``ssa_field_version`` cell when lane C posted
                # one is reached through that binding).
                self._bind(
                    value_id, incumbent, BindingKind.FIELD_INCUMBENT,
                    self._binding_cell(field_value_id),
                )
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
                # SEQUENCE_LENGTH: the sequence's contract cell is the cause.
                self._bind(
                    value_id, value, BindingKind.SEQUENCE_LENGTH,
                    self._sequence_contract_cell(int(sequence_id)), length_cell,
                )
        if follow_aliases:
            seen: set[int] = set()
            while value_id in self.value_aliases and value_id not in seen:
                seen.add(value_id)
                value_id = int(self.value_aliases[value_id])
        if value_id in self.declared_parameter_only_ids:
            self.declared_parameter_only_ids.discard(value_id)
            self._declare_parameter(value_id, ParameterDeclaration.USED)
        value = self.external_values.get(value_id)
        if value is None:
            value = self._value_from_meta(value_id, dtype=dtype)
            # PROVISIONAL_ARGUMENT: a value minted from absence -- no region
            # has published the id yet -- is recorded as unresolved; the
            # producer's later claim is the revision that resolves it.
            self._bind_unresolved(value_id, NO_PRODUCER_AT_USE, value=value)
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
            # REFINED: the value's row is revised to the demanded dtype and
            # the binding derives from that revision.
            previous_binding = self._binding_cell(value_id)
            self.ssa_value_objects[int(refined.id)] = refined
            revised = self._revise_value(refined, previous_binding)
            self._bind(
                value_id, refined, BindingKind.REFINED, previous_binding, revised,
            )
            self.arguments[:] = [
                refined if item is value else item for item in self.arguments
            ]
            value = refined
        return value

    def _sequence_contract_cell(self, sequence_id: int) -> Ref | None:
        """The ``sequence_contract_concordance`` cell of a resident sequence
        under this function's scopes, if committed."""

        book = self._book()
        page = book.pages.get("sequence_contract_concordance")
        if page is None:
            return None
        for scope in (
            self.function_name, str(self.tensor_shape_concordance_scope),
            self.lexical_read_scope,
        ):
            if scope is None:
                continue
            cell = self._declared_cell(
                "sequence_contract_concordance", (scope, int(sequence_id)),
            )
            if cell is not None:
                return cell
        return None

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
        cause: tuple = (),
    ) -> SSAValue:
        """The SSA value a definition of graph id ``value_id`` writes.

        ``cause`` names the cells of the definition being emitted (its
        operands): a claim that resolves a provisional binding derives from
        them, so the resolution has the cause the api asks for.
        """

        value_id = int(value_id)
        value = self.external_values.get(value_id)
        # The region that publishes this id, when one does: its
        # ``region_signature`` cell is the cause of a produced binding.
        producer = self._cells(self._producing_region_cell(value_id), *cause)
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
                    # REGION_RESULT: the claim resolves the PROVISIONAL row.
                    revised = self._revise_value(value, *producer)
                    self._bind(
                        value_id, value, BindingKind.REGION_RESULT,
                        *producer, revised,
                    )
                    return value
                # A preallocated arena is commonly both the initial value
                # entering control and the destination published by a later
                # region.  SSA versions the write; it is not an identity
                # conflict.  The source value ID stays in accounting so the
                # public arena-address policy can rotate the two versions.
                previous = value
                value = self.fresh_value(
                    dtype=dtype or value.dtype,
                    shape=tuple(value.shape),
                    transform=VERSIONED_WRITE,
                    operands=(self._binding_cell(value_id), previous),
                )
                value.accounting.update({
                    **dict(self.external_values[value_id].accounting or {}),
                    "source_value_id": value_id,
                    **(
                        {"ssa_inout_write_version": True}
                        if value_id in self.inout_value_ids else {}
                    ),
                })
                self._bind(
                    value_id, value, BindingKind.VERSIONED_WRITE, *producer,
                )
            return value
        value = self._value_from_meta(value_id, dtype=dtype)
        self._bind(value_id, value, BindingKind.REGION_RESULT, *producer)
        return value

    def _producing_region_cell(self, value_id: int) -> Ref | None:
        """The ``region_signature`` cell of the region whose outputs name
        ``value_id``, when one is on the book."""

        for region_index, (_feeds, outputs) in self.region_signatures.items():
            if int(value_id) in outputs:
                return self._declared_cell(
                    REGION_SIGNATURE.name, (self._scope(), int(region_index)),
                )
        return None

    def constant_value(self, literal: int, *operands: Any) -> SSAValue:
        """An int literal the builder needs: CONTROL_CONST from ``operands``
        (the cells of what the literal indexes or bounds; a bare literal has
        no identity cell of its own and names the function root)."""

        value = self.fresh_value(
            dtype="int", transform=CONTROL_CONST, operands=operands,
        )
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
        indices = (
            tuple(index)
            if isinstance(index, (tuple, list))
            else (index,)
        )
        row_shape = tuple(map(
            int,
            (source.accounting or {}).get("sequence_row_shape") or (),
        ))
        if row_shape and len(indices) == 1:
            # A shaped sequence column is one flat arena. Selecting row i is
            # an address view at i * elements_per_row; it neither loads one
            # scalar nor materializes a second tensor object.
            linear_index = indices[0]
            row_width = math.prod(row_shape)
            if row_width != 1:
                offset = self.fresh_value(
                    dtype="int64", transform=CONTROL_EXPRESSION,
                    operands=(linear_index, source),
                )
                self.emit(
                    Handler.Mul,
                    [linear_index, self.constant_value(int(row_width), source)],
                    offset,
                    attributes={
                        **attributes,
                        "binding": "shaped_sequence_row_offset",
                        "row_shape": row_shape,
                    },
                )
                linear_index = offset
            result = self.produced_value(
                result_id,
                dtype=str(source.dtype or "unknown"),
                claim_provisional_definition=claim_provisional_definition,
                cause=(source, linear_index),
            )
            result.shape = row_shape
            self._revise_value(result, source)
            self.emit(
                Handler.GetElementPtr,
                [source, linear_index],
                result,
                attributes={
                    **attributes,
                    "binding": "shaped_sequence_row_view",
                    "row_shape": row_shape,
                },
            )
            return result
        address = self.fresh_value(
            dtype="ptr", transform=ADDRESS, operands=(source, *indices),
        )
        self.emit(
            Handler.GetElementPtr,
            [source, *indices],
            address,
            attributes=attributes,
        )
        # A projection out of an ``ssa.aggregate`` handle is the region's
        # declared value, not the handle: the aggregate's dtype on a region
        # output typed a function's return merge as an aggregate, and the
        # caller's native result contract then returned a handle with no
        # scalar buffer (the managed window's ``dt_next`` read 0).
        dtype = str(source.dtype or "unknown")
        if dtype == "ssa.aggregate":
            declared = self._value_from_meta(int(result_id))
            dtype = str(declared.dtype or "unknown")
        result = self.produced_value(
            result_id,
            dtype=dtype,
            claim_provisional_definition=claim_provisional_definition,
            cause=(address,),
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
            handle_address = self.fresh_value(
                dtype="ptr", transform=ADDRESS, operands=(source, induction),
            )
            handle = self.fresh_value(
                dtype="int", transform=LOAD, operands=(handle_address,),
            )
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
            sequence_offset = self.fresh_value(
                dtype="int", transform=CONTROL_EXPRESSION,
                operands=(handle, sequence_stride),
            )
            row_index = self.fresh_value(
                dtype="int", transform=CONTROL_EXPRESSION,
                operands=(sequence_offset, induction),
            )
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
        offset = self.fresh_value(
            dtype="int", transform=CONTROL_EXPRESSION,
            operands=(row_index if selected_sequence else handle, row_stride),
        )
        row_base = self.fresh_value(
            dtype=str(child_data.dtype or "unknown"),
            transform=ROW_COLUMN_PROJECTION, operands=(child_data, offset),
        )
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
        self._bind(
            int(target_id), row_base, BindingKind.SEQUENCE_RESULT,
            child_data, offset,
        )
        return row_base

    def _nested_child_storage(
        self,
        child_key: tuple[str, int, int],
        target_id: int,
    ) -> tuple[SSAValue, SSAValue, SSAValue, SSAValue]:
        child = self.nested_child_rows.get(child_key)
        if child is None:
            target_meta = self.region_value_meta.get(int(target_id))
            # The child arena's four formals descend from the nested source
            # (``child_key[1]``, a graph id) and the target they serve.
            child_operands = (int(child_key[1]), int(target_id))
            child_data = self.fresh_value(dtype=(
                None if target_meta is None else str(target_meta.dtype)
            ), transform=DESCRIPTOR_CELL, operands=child_operands)
            child_lengths = self.fresh_value(
                dtype="int", transform=DESCRIPTOR_CELL, operands=(child_data,),
            )
            sequence_stride = self.fresh_value(
                dtype="int", transform=DESCRIPTOR_CELL, operands=(child_data,),
            )
            row_stride = self.fresh_value(
                dtype="int", transform=DESCRIPTOR_CELL, operands=(child_data,),
            )
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

    def _note_callsite_arguments(
        self, callsite_id: int, argument_ids, arguments,
    ) -> None:
        """Record what one scheduled callsite resolved its arguments to.

        A graph id that resolves to itself inside a loop body means the
        carried machinery did not map it to the current iteration.
        """

        book = self._book()
        for position, (graph_id, value) in enumerate(
            zip(argument_ids, arguments)
        ):
            row = (str(self.function_name), int(callsite_id), position)
            fact = (int(graph_id), int(value.id), bool(self.loop_targets))
            # DERIVED(the ``call_argument_operand`` cell of this position,
            # the resolved binding cell of the argument).
            sources = self._cells(
                self._declared_cell(
                    "call_argument_operand",
                    (self.lexical_read_scope, int(callsite_id), int(position)),
                ) if self.lexical_read_scope is not None else None,
                self._binding_cell(int(graph_id)), value,
            )
            latest = book.latest_ref(CALLSITE_ARGUMENT, row)
            if latest is not None and (
                book.pages[CALLSITE_ARGUMENT.name].cells.get(
                    (row, latest.column)
                ) == fact
            ):
                continue
            book.post(
                CALLSITE_ARGUMENT, row, fact,
                stage=self.active_stage,
                provenance=Derived(sources),
                mode=Mode.REVISE,
            )

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
            self._callsite_argument(int(callsite_id), position, int(value_id))
            for position, value_id in enumerate(argument_ids)
        ]
        self._note_callsite_arguments(callsite_id, argument_ids, arguments)
        for index, value in enumerate(arguments):
            if self._value_dominates_current_edge(value):
                continue
            # A source-call record can retain an arm's assigned value ID
            # after the conditional has published its merge. Only a unique
            # dominating Phi that explicitly consumes that value can repair
            # the binding; unrelated live values or names are not evidence.
            merges = {
                int(instruction.res.id): instruction.res
                for emitted in self.blocks.values()
                for instruction in emitted.instrs
                if instruction.op == "Phi"
                and instruction.res is not None
                and instruction.attributes.get("binding") == "conditional_carried"
                and any(argument is value for argument in instruction.args)
                and self._value_dominates_current_edge(instruction.res)
            }
            if len(merges) == 1:
                arguments[index] = next(iter(merges.values()))
            elif len(merges) > 1:
                self.shortfalls.append(SSALoweringShortfall(
                    "control", "ambiguous-call-conditional-merge", location,
                    f"callsite {callsite_id} argument {value.id} has multiple "
                    f"dominating conditional merges {sorted(merges)}",
                ))
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
            aggregate = self.fresh_value(
                dtype="ssa.aggregate", transform=REGION_CALL_RESULT,
                operands=tuple(arguments),
            )
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
                address = self.fresh_value(
                    dtype="ptr", transform=ADDRESS, operands=(aggregate,),
                )
                self.emit(
                    Handler.GetElementPtr,
                    [aggregate, self.constant_value(output_index, aggregate)],
                    address,
                    attributes=projection_attributes,
                )
                # The projection's dtype/shape are the region-declared facts
                # of the caller result, not the aggregate handle's dtype.
                result = self.produced_value(int(output_id), cause=(address,))
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
                # CALL_RESULT: the callsite's argument cells are the cause.
                self._bind(
                    primary, result, BindingKind.CALL_RESULT, *arguments,
                )
        self.emit(
            Handler.Call,
            arguments,
            result,
            attributes=marker_attributes,
        )
        self.emitted_plan_callsites.add(int(callsite_id))

    # -- reads resolved through the identity book --------------------------
    #
    # One base fact decides every read: ``lexical_read_binding`` row
    # ``(read scope, consumer, role, ordinal)`` names the authored binding an
    # operand read.  Planner pages record only WHICH operand a lowering
    # operand is (``consumer_operand``, ``call_argument_operand``,
    # ``region_feed_consumer``); a loop's entry state records which bindings
    # it carries under each initial id and their pre-loop value
    # (``loop_entry_state``).  The builder keeps only its position: the
    # loops enclosing the current emission point.

    def _book(self) -> Any:
        from .identity_concordance import current_identity_book

        return current_identity_book()

    # ------------------------------------------- the book's api, step 5
    # Every SSA value this builder mints is a NOVEL ``ssa_value`` row; every
    # graph id it binds is a ``control_value_binding`` revision DERIVED from
    # the cells that caused it (plan 80, part B).  ``external_values`` and
    # ``ssa_value_objects`` are the read views: the dict readers keep, and
    # the object table the emitted instructions hold (objects are mutated
    # in place after creation, so the book holds facts and the table holds
    # the objects, keyed by the book's id).

    def _scope(self) -> Any:
        """The function scope the builder's pages are keyed by: the control
        scope of THIS lowering (book-numbered by ``mint_scope``).

        Not the graph's ``lexical_read_scope``: one ControlProgram is lowered
        more than once while specializations settle (see
        ``_enter_loop_state``), and rows whose facts are this lowering's
        physical cells (``carried_snapshot``, ``function_output``, ...) must
        not be concorded against another lowering's.  Graph-side cells
        (``canonical_value``, ``name_binding``) are read under
        ``lexical_read_scope`` as before.
        """

        return str(self.tensor_shape_concordance_scope)

    def _stage(self, stage: Any = None) -> Any:
        return stage if stage is not None else self.active_stage

    def _function_root(self) -> Ref:
        """The one root row of this lowering: ``cell_set`` row ``(scope, 0)``
        posted NOVEL(CONTROL_FUNCTION_ROOT) with no NEW, so it mints nothing
        and carries only its origin edge.  A mint that has no more specific
        cell to name (a literal, a function the reducer never saw) names it.
        """

        if self._root_cell is None:
            self._root_cell = self._book().post(
                CELL_SET, (self._scope(), 0), (),
                stage=CONTROL_SSA_ENTRY,
                provenance=Novel(CONTROL_FUNCTION_ROOT, ()),
                mode=Mode.CONCORD,
            )
        return self._root_cell

    def _cell_set(self, cells: tuple, stage: Any = None) -> Ref:
        """One ``cell_set`` row DERIVED from every cell in ``cells``; its
        fact repeats them as keys.  The operand of a mint made from several
        cells (the api's transforms take one operand)."""

        book = self._book()
        scope = self._scope()
        ordinal = book.page(CELL_SET).scope_row_count(scope) + 1
        return book.post(
            CELL_SET, (scope, ordinal), tuple(cell.key for cell in cells),
            stage=self._stage(stage),
            provenance=Derived(tuple(cells)),
            mode=Mode.CONCORD,
        )

    def _canonical_cell(self, graph_id: Any) -> Ref | None:
        """The ``canonical_value`` cell of graph id ``graph_id``, if the
        reducer saw this function."""

        if self.lexical_read_scope is None or graph_id is None:
            return None
        row = (self.lexical_read_scope, int(graph_id))
        return self._book().latest_ref(CANONICAL_VALUE, row) or (
            self._book().latest_ref(INGESTION_VALUE, row)
        )

    def _declared_cell(self, page_name: str, row: tuple) -> Ref | None:
        """Read-side lookup of a cell on a page another lane declares
        (``item_operand``, ``consumer_operand``, ...): None until the page
        is declared and the cell exists."""

        book = self._book()
        page = book.registry.pages.get(page_name)
        if page is None:
            return None
        return book.latest_ref(page, row)

    def _cells(self, *items: Any) -> tuple[Ref, ...]:
        """Cells named loosely -- a Ref, an SSAValue (its ``ssa_value``
        cell), a graph id (its ``canonical_value`` cell) or None -- as the
        distinct Refs that exist."""

        found: list[Ref] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, Ref):
                cell = item
            elif isinstance(item, SSAValue):
                cell = self._value_cell(item)
            elif isinstance(item, int) and not isinstance(item, bool):
                cell = self._canonical_cell(item)
            else:
                continue
            if cell is not None and cell not in found:
                found.append(cell)
        return tuple(found)

    def _value_cell(
        self, value: SSAValue, *cells: Any, stage: Any = None,
    ) -> Ref:
        """The ``ssa_value`` cell of ``value``.

        A value the builder minted has its row already.  A value whose id
        IS a graph id (``_value_from_meta``, a region's declared value) is
        adopted on first sight: its row is DERIVED from the id's
        ``canonical_value`` cell (and ``cells``), or posted
        ``Unsourced(GRAPH_ID_WITHOUT_CANONICAL_CELL)`` when the reducer
        never saw this function (plan 80 B6 R8).
        """

        book = self._book()
        row = (self._scope(), int(value.id))
        ref = book.latest_ref(SSA_VALUE, row)
        if ref is not None:
            # The object most recently seen under the id is the one the
            # emitted instructions hold (a produced value replaces the
            # object its mint handed out).
            self.ssa_value_objects[int(value.id)] = value
            return ref
        fact = SSAValueFact(
            value.dtype, tuple(value.shape or ()),
            SSAValueOrigin.ADOPTED_GRAPH_ID,
        )
        sources = self._cells(self._canonical_cell(value.id), *cells)
        ref = book.post(
            SSA_VALUE, row, fact,
            stage=self._stage(stage),
            provenance=(
                Derived(sources) if sources
                else Unsourced(GRAPH_ID_WITHOUT_CANONICAL_CELL)
            ),
            mode=Mode.REVISE,
        )
        self.ssa_value_objects[int(value.id)] = value
        return ref

    def fresh_value(
        self,
        *,
        dtype: str | None = None,
        shape: tuple[int, ...] = (),
        transform: Any,
        operands: tuple = (),
        stage: Any = None,
    ) -> SSAValue:
        """Mint one SSA value through the book: NOVEL(``transform``,
        operand) on ``ssa_value`` row ``(scope, NEW)``.

        ``operands`` name what the value was made from (Refs, SSAValues or
        graph ids, see ``_cells``); several cells become one ``cell_set``
        row, none becomes the function root.  The id is the book's.
        """

        stage = self._stage(stage)
        cells = self._cells(*operands)
        if len(cells) == 1:
            operand = cells[0]
        elif cells:
            operand = self._cell_set(cells, stage)
        else:
            operand = self._function_root()
        ref = self._book().post(
            SSA_VALUE, (self._scope(), NEW),
            SSAValueFact(dtype, tuple(shape), SSAValueOrigin.MINTED),
            stage=stage,
            provenance=Novel(transform, (operand,)),
            mode=Mode.REVISE,
        )
        value = SSAValue(int(ref.row[1]), dtype=dtype, shape=shape)
        self.ssa_value_objects[int(value.id)] = value
        return value

    def _revise_value(
        self, value: SSAValue, *cells: Any, stage: Any = None,
    ) -> Ref | None:
        """Record an in-place dtype/shape change of ``value`` as a REVISE
        of its ``ssa_value`` row DERIVED(previous cell, ``cells``)."""

        book = self._book()
        row = (self._scope(), int(value.id))
        previous = book.latest_ref(SSA_VALUE, row)
        if previous is None:
            return self._value_cell(value, *cells, stage=stage)
        recorded = book.pages[SSA_VALUE.name].cells.get((row, previous.column))
        origin = getattr(recorded, "origin", SSAValueOrigin.ADOPTED_GRAPH_ID)
        fact = SSAValueFact(value.dtype, tuple(value.shape or ()), origin)
        if fact == recorded:
            return previous
        return book.post(
            SSA_VALUE, row, fact,
            stage=self._stage(stage),
            provenance=Derived(self._cells(previous, *cells)),
            mode=Mode.REVISE,
        )

    def _binding_cell(self, graph_id: Any) -> Ref | None:
        return self._book().latest_ref(
            CONTROL_VALUE_BINDING, (self._scope(), int(graph_id)),
        )

    def _binding_fact(self, graph_id: Any) -> Any:
        cell = self._binding_cell(graph_id)
        if cell is None:
            return None
        page = self._book().pages.get(CONTROL_VALUE_BINDING.name)
        return None if page is None else page.cells.get((cell.row, cell.column))

    def _binding(self, graph_id: Any) -> SSAValue | None:
        """The SSA value bound to ``graph_id`` (the read view)."""

        return self.external_values.get(int(graph_id))

    def _bind(
        self,
        graph_id: Any,
        value: SSAValue,
        kind: BindingKind,
        *cells: Any,
        stage: Any = None,
    ) -> Ref | None:
        """Bind graph id ``graph_id`` to ``value``: one REVISE row on
        ``control_value_binding`` DERIVED(the id's ``canonical_value`` cell,
        the value's ``ssa_value`` cell, ``cells``), then the read view.

        Rebinding the same value is a no-op (no post); a function the
        reducer never saw binds with kind UNSCOPED (R8).
        """

        graph_id = int(graph_id)
        self.external_values[graph_id] = value
        if value is None:
            return None
        book = self._book()
        value_cell = self._value_cell(value, stage=stage)
        row = (self._scope(), graph_id)
        latest = book.latest_ref(CONTROL_VALUE_BINDING, row)
        if latest is not None:
            recorded = book.pages[CONTROL_VALUE_BINDING.name].cells.get(
                (row, latest.column)
            )
            if (
                isinstance(recorded, ControlBinding)
                and recorded.ssa_value == value_cell
            ):
                return latest
        if self.lexical_read_scope is None:
            kind = BindingKind.UNSCOPED
        sources = self._cells(self._canonical_cell(graph_id), value_cell, *cells)
        return book.post(
            CONTROL_VALUE_BINDING, row, ControlBinding(value_cell, kind),
            stage=self._stage(stage),
            provenance=Derived(sources),
            mode=Mode.REVISE,
        )

    def _bind_unresolved(
        self,
        graph_id: Any,
        reason: Any,
        *cells: Any,
        value: SSAValue | None = None,
        stage: Any = None,
    ) -> Ref | None:
        """Record that ``graph_id`` has no resolvable binding here:
        ``Unresolved(reason, read=cells)`` on its binding row, DERIVED from
        what was read.  ``value`` (a stand-in) goes to the read view only."""

        graph_id = int(graph_id)
        if value is not None:
            self.external_values[graph_id] = value
        read = self._cells(self._canonical_cell(graph_id), *cells)
        if value is not None:
            read = self._cells(*read, self._value_cell(value, stage=stage))
        if not read:
            return None
        return self._book().post(
            CONTROL_VALUE_BINDING, (self._scope(), graph_id),
            Unresolved(reason, read=read),
            stage=self._stage(stage),
            provenance=Derived(read),
            mode=Mode.REVISE,
        )

    def _declare_parameter(
        self, graph_id: Any, declaration: ParameterDeclaration, *cells: Any,
    ) -> Ref | None:
        """``declared_parameter`` row ``(scope, graph id)``: CALL_ONLY
        DERIVED(the parameter's version-0 ``name_binding`` cell, its binding
        cell); USED DERIVED(the previous declaration, the use's cell)."""

        graph_id = int(graph_id)
        book = self._book()
        row = (self._scope(), graph_id)
        previous = book.latest_ref(DECLARED_PARAMETER, row)
        if previous is not None:
            recorded = book.pages[DECLARED_PARAMETER.name].cells.get(
                (row, previous.column)
            )
            if recorded is declaration:
                return previous
        sources = self._cells(
            previous, *cells, self._binding_cell(graph_id),
            self._canonical_cell(graph_id),
        )
        if not sources:
            return None
        return book.post(
            DECLARED_PARAMETER, row, declaration,
            stage=self.active_stage,
            provenance=Derived(sources),
            mode=Mode.REVISE,
        )

    def _rebind_view(
        self, values: Mapping[int, SSAValue], kind: BindingKind, *cells: Any,
    ) -> None:
        """Rebind every id in ``values`` whose current view differs (the
        ``update(...)`` writers: restores and merges)."""

        for graph_id, value in values.items():
            if self.external_values.get(int(graph_id)) is value:
                continue
            self._bind(int(graph_id), value, kind, *cells)

    def _withdraw(self, graph_id: Any, previous: SSAValue | None, *cells: Any) -> None:
        """``pop`` then ``[...] = previous``: the saved value is rebound
        RESTORED; with no previous the binding is withdrawn with an
        ``Unresolved(BINDING_WITHDRAWN)`` revision, never an erasure."""

        graph_id = int(graph_id)
        current = self.external_values.pop(graph_id, None)
        if previous is None:
            if current is not None:
                self._bind_unresolved(
                    graph_id, BINDING_WITHDRAWN, *cells, self._value_cell(current),
                )
            return
        self._bind(graph_id, previous, BindingKind.RESTORED, *cells)

    def _construct_cell(self, node_id: Any) -> Ref | None:
        """The identity cell of a control construct's source graph node."""

        return self._canonical_cell(node_id)

    def _restore_view(self, saved: Mapping[int, SSAValue], *cells: Any) -> None:
        """Make the read view equal ``saved`` again (an arm that leaves
        through its own edge): every id the arm added is withdrawn, every
        id it rebound is RESTORED.  Each change is a revision, never an
        erasure."""

        for graph_id in tuple(self.external_values):
            if int(graph_id) not in saved:
                self._withdraw(int(graph_id), None, *cells)
        for graph_id, value in saved.items():
            if self.external_values.get(int(graph_id)) is not value:
                self._bind(int(graph_id), value, BindingKind.RESTORED, *cells)

    def _post_carried_snapshots(
        self, conditional_cell: Ref | None, snapshots: Mapping[int, SSAValue],
    ) -> dict[int, Ref]:
        """One ``carried_snapshot`` row per snapshotted initial id: the
        binding cell current at branch entry, DERIVED from that cell and
        the conditional construct's cell.  Returns initial id -> the cell
        that stands for the snapshot (the row's cell; the binding cell when
        the conditional has no identity cell to key the row by)."""

        book = self._book()
        cells: dict[int, Ref] = {}
        for initial_id, value in snapshots.items():
            initial_id = int(initial_id)
            binding_cell = self._binding_cell(initial_id)
            if binding_cell is None:
                binding_cell = self._value_cell(value)
            if conditional_cell is None:
                cells[initial_id] = binding_cell
                continue
            row = (self._scope(), conditional_cell, initial_id)
            existing = book.latest_ref(CARRIED_SNAPSHOT, row)
            if existing is not None:
                cells[initial_id] = existing
                continue
            cells[initial_id] = book.post(
                CARRIED_SNAPSHOT, row, binding_cell,
                stage=CONTROL_SSA_CONDITIONAL,
                provenance=Derived(self._cells(binding_cell, conditional_cell)),
                mode=Mode.CONCORD,
            )
        return cells

    def _concorded_resident(self, value_id: int) -> int:
        """The resident ``value_id`` denotes under the CONCORDED planning
        aliases (the ``control_value_alias`` PLANNING rows as resolved at
        builder entry), never the live ``value_aliases`` view that loops
        rewrite while their bodies lower."""

        current = int(value_id)
        seen: set[int] = set()
        while current in self.concorded_value_aliases:
            if current in seen:
                raise ValueError(
                    "cyclic durable planning concordance for "
                    f"{self.function_name!r}: {current}"
                )
            seen.add(current)
            target = int(self.concorded_value_aliases[current])
            if target == current:
                break
            current = target
        return current

    def _recorded_name_write(self, value_id: int) -> bool:
        """Whether the book records an authored write producing ``value_id``:
        a ``name_binding`` row under the read scope whose fact names it.

        This is the record decision 7.2 asks for: an arm id with no such
        row is not a write the arm made (a merge node, a planner alias), so
        the arm takes the entered version; an arm id with one and no
        binding in this lowering is a version the builder cannot find.

        An in-place store (``m[2] = ...``, ``m[:2] -= ...``) is the one
        version that has BOTH: an authored ``name_binding`` row and a
        PLANNING alias to the storage it writes (store versions are versions
        of one storage).  The builder never binds such a version, so
        ``_carried_name_arm`` resolves the arm through the concorded alias
        chain before asking this question; this answer alone never decides
        an in-place store arm.
        """

        if self.lexical_read_scope is None:
            return False
        recorded = getattr(self, "_authored_write_ids", None)
        if recorded is None:
            recorded = set()
            page = self._book().pages.get(NAME_BINDING.name)
            if page is not None:
                for row in page.scope_rows(self.lexical_read_scope):
                    fact = page.latest(row)
                    value = getattr(fact, "value_id", None)
                    if value is not None and getattr(fact, "authored", True):
                        recorded.add(int(value))
            self._authored_write_ids = recorded
        return int(value_id) in recorded

    def _carried_name_arm(
        self,
        arm_id: Any,
        initial_id: Any,
        snapshot: SSAValue,
        snapshot_cell: Ref | None,
        *,
        path: str,
    ) -> tuple[SSAValue, Any]:
        """The SSA value a name-carried merge takes from one arm, and the
        cell it derives from (decision 7.2 of the design).

        The arm id equal to the initial id: the arm did not rebind the name,
        the entered version (the snapshot) is the arm and the snapshot cell
        is its source.  A different arm id with a binding: that binding (the
        version the book records for the arm).  A different arm id with no
        binding: the graph records a version the builder cannot find --
        ``Unresolved(NAME_ARM_VERSION_MISSING)`` is posted at the arm id, a
        ``carried-name-arm-missing`` shortfall is appended, and the snapshot
        stands in for the emitted Phi without being posted as the arm.
        """

        arm_id = int(arm_id)
        if arm_id == int(initial_id):
            return snapshot, snapshot_cell
        value = self.external_values.get(arm_id)
        if value is not None:
            source = self._binding_cell(arm_id)
            return value, source if source is not None else self._value_cell(value)
        if arm_id in self.constant_value_ids:
            # An authored literal (``rejected = False`` in the arm): no
            # region publishes it; the control function owns it.  Resolve it
            # as the provisional formal ``_materialize_control_constants``
            # turns into this function's own ``Const`` -- as a folded
            # ``flag = True; break`` is resolved on its break edge and an
            # owned literal output is resolved at the exit.
            value = self.external_value(arm_id)
            source = self._binding_cell(arm_id)
            return value, source if source is not None else self._value_cell(value)
        # An in-place store version is aliased (``control_value_alias``,
        # PLANNING) to the storage it writes and is never bound.
        # ``m = call(); if c: m[2] = ...; if t > 0: m[:2] -= ...`` merges the
        # inner ``if`` with arm = the ``m[:2]`` store and initial = the
        # ``m[2]`` store: both resolve to the call result, so the arm
        # carries the entered storage.  Resolve through the CONCORDED alias
        # chain (not the live view loops rewrite) before the authored-write
        # check; refuse only a distinct resolved version with no binding.
        resident = self._concorded_resident(arm_id)
        if resident == self._concorded_resident(initial_id):
            return snapshot, snapshot_cell
        if resident != arm_id:
            value = self.external_values.get(resident)
            if value is not None:
                source = self._binding_cell(resident)
                return value, (
                    source if source is not None else self._value_cell(value)
                )
        if not self._recorded_name_write(arm_id):
            # The book records no authored write of this value in the arm
            # (no ``name_binding`` row names it): the arm did not assign, the
            # value falls up the scope ladder to the entered version.  Never
            # refused (design decision 7.2).
            return snapshot, snapshot_cell
        missing = self._bind_unresolved(
            arm_id, NAME_ARM_VERSION_MISSING, snapshot_cell,
        )
        self.shortfalls.append(SSALoweringShortfall(
            "control", "carried-name-arm-missing", path,
            "a name-carried conditional arm names a version with no "
            f"binding in this lowering: arm={arm_id} initial={int(initial_id)}"
            + ("" if missing is None else f" (recorded {NAME_ARM_VERSION_MISSING.name})"),
        ))
        return snapshot, None

    # ------------------------------------------------- field-state versions
    # A scalar field write and a conditional field merge each ARE one
    # ``reducer_field_state`` cell (plan 70, sections 4-5).  The SSA value
    # this builder emits for such a cell is published on ``ssa_field_version``
    # at ``(function scope, cell)``; the conditional merge reads each arm's
    # version at the arm's own cell.  ``external_values[<graph id>]`` stays
    # as a read view, never as the source of an arm.

    def _field_version_row(self, cell: Any) -> tuple | None:
        if self.lexical_read_scope is None or not isinstance(cell, Ref):
            return None
        return (self.lexical_read_scope, cell)

    def _publish_field_version(
        self, cell: Any, value: SSAValue, sources: tuple[Any, ...],
    ) -> Any:
        """Post ``value`` as the SSA version of field-state ``cell``.

        ``sources`` are the cells the version derives from (the field-state
        cell itself for a write; the arm versions and the merged cell for a
        merge).  Returns the version cell's Ref, or None when the function
        has no reduction scope to key the row by.
        """
        row = self._field_version_row(cell)
        if row is None:
            return None
        ref = self._book().post(
            SSA_FIELD_VERSION, row, int(value.id),
            stage=CONTROL_SSA,
            provenance=Derived(tuple(
                source for source in sources if isinstance(source, Ref)
            )),
            mode=Mode.CONCORD,
        )
        self.field_version_values[ref] = value
        return ref

    def _field_state_kind(self, cell: Ref) -> Any:
        """The ``FieldStateKind`` recorded at a ``reducer_field_state`` cell."""
        page = self._book().pages.get(cell.page.name)
        fact = None if page is None else page.cells.get((cell.row, cell.column))
        return getattr(fact, "kind", None)

    def _carried_field_arm(
        self,
        arm_cell: Any,
        other_arm_cell: Any,
        snapshot: SSAValue,
        *,
        path: str,
        arm_value_id: Any = None,
    ) -> tuple[SSAValue, Any]:
        """The SSA value a field-carried merge takes from one arm.

        Returns ``(value, source)`` where ``source`` is the cell the merged
        version derives from for this arm: the arm's ``ssa_field_version``
        cell when the arm wrote, or the arm cell itself when the arm did not
        write (the arm cell IS the pre-branch cell -- an OBSERVED state, or
        the same cell as the other arm -- so the snapshot is its value).  An
        arm cell that names a write or a merge with no version posted is
        recorded as ``Unresolved(ARM_VERSION_MISSING)`` at its row and a
        shortfall is raised; the snapshot then stands in for the emitted Phi
        but is never posted as the arm.

        ``arm_value_id`` is the arm's graph value the planner paired with
        ``arm_cell`` (the reducer Phi's arm parent; ``carried_aliases`` and
        ``carried_field_cells`` are index-aligned).  An ELEMENT_WRITTEN cell
        (``obj.field[i] = v``, plan 70 section 1.3) holds the IndexedStore
        as both value and effect; that store is emitted inside a planned
        region, so no block of this builder publishes a version for it.
        Its version IS the storage the store writes in place: the store id
        is aliased to that storage by its ``control_value_alias`` PLANNING
        row (the same chain ``_carried_name_arm`` resolves an in-place
        store through), and the storage's current binding is the value.
        Program: ``dt_controller.step_with_dt_control_used``, ``if
        rejected: metrics.control_values[0] = ...; metrics.control_present[0]
        = 1.0`` on the retry loop -- the only statements of the arm are
        element writes, and the merge named their ELEMENT_WRITTEN cells
        with no version (``carried-field-arm-missing``, N=2 orbital dt
        system; tools/compiler_probes/probe_element_written_field_arm.py).
        """
        book = self._book()
        row = self._field_version_row(arm_cell)
        if row is None:
            return snapshot, None
        version_ref = book.latest_ref(SSA_FIELD_VERSION, row)
        if version_ref is not None:
            fact = book.pages[SSA_FIELD_VERSION.name].latest(row)
            value = self.field_version_values.get(version_ref)
            if value is None and isinstance(fact, int):
                # The version was posted by another builder over this scope;
                # the row's fact IS the SSA identity, so name it directly.
                value = SSAValue(
                    int(fact), dtype=snapshot.dtype, shape=snapshot.shape,
                )
            if value is not None:
                return value, version_ref
        kind = self._field_state_kind(arm_cell)
        if arm_cell == other_arm_cell or kind in {
            FieldStateKind.OBSERVED, FieldStateKind.ARM_SELECTED,
            FieldStateKind.LOOP_EXIT,
        }:
            # The arm did not write: its cell is the state the branch was
            # entered with, whose SSA value is the snapshot.
            return snapshot, arm_cell
        if kind is FieldStateKind.ELEMENT_WRITTEN and arm_value_id is not None:
            storage_id = self._concorded_resident(int(arm_value_id))
            storage = self.external_values.get(storage_id)
            if storage is not None:
                # ELEMENT_WRITTEN version: the in-place storage, DERIVED(the
                # ELEMENT_WRITTEN cell, the store's PLANNING alias cell, the
                # storage's binding cell).
                version_ref = self._publish_field_version(
                    arm_cell, storage, (
                        arm_cell,
                        book.latest_ref(
                            CONTROL_VALUE_ALIAS,
                            (self._scope(), int(arm_value_id)),
                        ),
                        self._binding_cell(storage_id),
                    ),
                )
                return storage, (
                    version_ref if version_ref is not None else arm_cell
                )
        missing = book.post(
            SSA_FIELD_VERSION, row,
            Unresolved(ARM_VERSION_MISSING, read=(arm_cell,)),
            stage=CONTROL_SSA,
            provenance=Derived((arm_cell,)),
            mode=Mode.REVISE,
        )
        self.shortfalls.append(SSALoweringShortfall(
            "control", "carried-field-arm-missing", path,
            "a field-carried conditional arm names a field-state cell with "
            f"no ssa_field_version: row={missing.row!r} "
            f"(recorded {ARM_VERSION_MISSING.name})",
        ))
        return snapshot, None

    def _operand_bindings(self, operands: Any) -> set:
        """The bindings read by ``(consumer, role, ordinal)`` operands.

        An ``Unresolved`` revision (a withdrawn position) reads as absence.
        """

        page = self._book().page(LEXICAL_READ_BINDING)
        found = set()
        for consumer, role, ordinal in operands:
            fact = page.latest((self.lexical_read_scope, consumer, role, ordinal))
            found.add(None if isinstance(fact, Unresolved) else fact)
        return found

    def _read_binding_cells(self, operands: Any) -> tuple[Ref, ...]:
        """The ``lexical_read_binding`` cells at ``(consumer, role,
        ordinal)`` operand positions that have one."""

        if self.lexical_read_scope is None:
            return ()
        book = self._book()
        return self._cells(*(
            book.latest_ref(
                LEXICAL_READ_BINDING,
                (self.lexical_read_scope, consumer, role, ordinal),
            )
            for consumer, role, ordinal in operands
        ))

    def _enter_loop_state(self, loop: Any, carried: Any) -> Any:
        """Commit this loop's entry state; return its key for the stack.

        Row ``(control scope, loop node, initial id)`` holds the bindings the
        loop carries under that initial id (``loop_carried_binding``) and
        the value that initial id had before the loop.  A pair the book does
        not attribute is not committed; its reads keep the header rebinding.
        """

        loop_node = getattr(loop, "source_loop_node_id", None)
        if self.lexical_read_scope is None or loop_node is None:
            return None
        book = self._book()
        loop_cell = self._construct_cell(loop_node)
        key = (str(self.tensor_shape_concordance_scope), int(loop_node))
        states: dict[int, set] = {}
        state_sources: dict[int, list] = {}
        for entry, (updated_id, initial_id, initial_value, *_rest) in (
            enumerate(carried)
        ):
            binding_row = (
                self.lexical_read_scope, int(loop_node),
                int(updated_id), int(initial_id),
            )
            binding_cell = book.latest_ref(LOOP_CARRIED_BINDING, binding_row)
            bindings = book.page(LOOP_CARRIED_BINDING).latest(binding_row)
            if bindings is None or isinstance(bindings, Unresolved):
                # A carried pair the book does not attribute: recorded, not
                # silently kept as the header rebinding (plan 80 B2.4).
                self._bind_unresolved(
                    int(initial_id), CARRIED_ENTRY_NOT_ATTRIBUTED,
                    self._binding_cell(int(updated_id)), loop_cell,
                )
                continue
            owned = states.setdefault(int(initial_id), set())
            owned.update(bindings)
            state_sources.setdefault(int(initial_id), []).append(binding_cell)
            # Row ``(control scope, loop, binding)`` -> the carried entry
            # that binding is.  Two bindings seeded from one value
            # (``second = value; third = value``) share an initial id but
            # are distinct entries with distinct header Phis.  DERIVED(the
            # ``loop_carried_binding`` cell it read).
            for binding in bindings:
                self._post_concord(
                    LOOP_CARRIED_ENTRY, (*key, str(binding)), int(entry),
                    binding_cell, loop_cell, stage=CONTROL_SSA_LOOP,
                )
        for initial_id, owned in states.items():
            # The book records the authored state identity, never a physical
            # SSAValue from one lowering.  One ControlProgram may be lowered
            # more than once while specializations settle; retaining the
            # first lowering's SSA object made the next lowering disagree on
            # incidental dtype/shape metadata for the same semantic row.
            # Its concrete pre-loop value is already owned by this builder's
            # loop frame and is recovered there by _resolve_read.  DERIVED(
            # the same cells, the pre-loop binding cell of the initial).
            self._post_concord(
                LOOP_ENTRY_STATE, (*key, initial_id),
                (tuple(sorted(owned)), int(initial_id)),
                *state_sources.get(initial_id, ()),
                self._binding_cell(initial_id), loop_cell,
                stage=CONTROL_SSA_LOOP,
            )
        self.loop_frames[key] = {"carried": carried, "latch": False}
        return key

    def _post_concord(
        self, page: Any, row: tuple, fact: Any, *cells: Any, stage: Any = None,
    ) -> Ref:
        """One CONCORD post DERIVED from ``cells`` (the function root when
        none exists), replacing a raw ``concord`` on a declared page."""

        sources = self._cells(*cells)
        if not sources:
            sources = (self._function_root(),)
        return self._book().post(
            page, row, fact,
            stage=self._stage(stage),
            provenance=Derived(sources),
            mode=Mode.CONCORD,
        )

    def _loop_scope_rebinds(self, loop: Any, carried: Any) -> tuple:
        """Describe every carried entry without collapsing shared seeds.

        The value triple alone is insufficient source identity: two authored
        bindings may share an outer value while selecting different updates.
        Carry the lexical binding receipt into the loop-scope declaration so
        the concordance can represent those simultaneous possibilities as
        separate rows.
        """

        loop_node = getattr(loop, "source_loop_node_id", None)
        page = self._book().page("loop_carried_binding")
        binding_scope = (
            None
            if self.lexical_read_scope is None or loop_node is None
            else (self.lexical_read_scope, int(loop_node))
        )
        return tuple(
            (
                int(initial.id), int(current.id), int(updated.id),
                int(initial_id), int(updated_id),
                tuple(
                    () if binding_scope is None else (
                        page.latest((
                            *binding_scope,
                            int(updated_id), int(initial_id),
                        )) or ()
                    )
                ),
            )
            for updated_id, initial_id, initial, updated, current in carried
        )

    def _resolve_read(
        self, value_id: int, bindings: set, reader: str,
    ) -> Any | None:
        """The value ``bindings`` see for ``value_id`` at this point.

        Walks the enclosing loops innermost first.  A loop that carries
        every read binding under this initial id is where the value is (its
        header, or its latch update -- the ordinary resolution); a loop
        carrying none of them is passed through to its pre-loop value; a
        loop carrying some, or a read no row attributes, raises.  Returns
        None when the ordinary resolution is the answer.
        """

        if not bindings:
            return None
        state_page = self._book().page("loop_entry_state")
        candidate = None
        for key in reversed(self.enclosing_loop_states):
            if key is None:
                continue
            state = state_page.latest((*key, int(value_id)))
            if state is None:
                continue
            carried = set(state[0])
            frame = self.loop_frames.get(key)
            before = None
            if frame is not None:
                before = next((
                    initial
                    for _updated_id, initial_id, initial, *_rest
                    in frame["carried"]
                    if int(initial_id) == int(value_id)
                ), None)
            if bindings <= carried:
                entry_value = self._carried_entry_value(
                    key, int(value_id), bindings, reader,
                )
                return candidate if entry_value is None else entry_value
            if None in bindings:
                raise ValueError(
                    f"{self.function_name}: {reader} reads carried initial "
                    f"{value_id} through an operand no lexical_read_binding "
                    "row attributes"
                )
            if bindings & carried:
                raise ValueError(
                    f"{self.function_name}: {reader} reads value {value_id} "
                    f"as bindings {sorted(bindings, key=repr)!r}, of which "
                    f"the enclosing loop carries only {sorted(carried)!r}"
                )
            if before is not None:
                candidate = before
        return candidate

    def _carried_entry_value(
        self, key: Any, value_id: int, bindings: set, reader: str,
    ) -> Any | None:
        """The carried entry's value for ``bindings`` read in loop ``key``.

        The entry is the book's (``loop_carried_entry``); its header Phi in
        the body, its published update while the latch re-evaluates the
        predicate.  When the read's initial id seeds one entry only, the
        ordinary resolution (the id's rebinding) already is that entry and
        None is returned.  Bindings naming two entries cannot be one read.
        """

        frame = self.loop_frames.get(key)
        if frame is None:
            return None
        carried = frame["carried"]
        sharing = [
            entry for entry, (_updated, initial_id, *_rest)
            in enumerate(carried) if int(initial_id) == value_id
        ]
        if len(sharing) <= 1:
            return None
        entry_page = self._book().page("loop_carried_entry")
        entries = {
            entry_page.latest((*key, str(binding))) for binding in bindings
        }
        if None in entries or len(entries) != 1:
            raise ValueError(
                f"{self.function_name}: {reader} reads carried initial "
                f"{value_id} as bindings {sorted(bindings, key=repr)!r}, "
                f"which name carried entries {sorted(entries, key=repr)!r}; "
                "one read is one entry"
            )
        updated_id, _initial_id, _initial, updated, current = carried[
            int(entries.pop())
        ]
        if frame["latch"]:
            return self.external_values.get(int(updated_id), updated)
        return current

    def _read(self, value_id: int, bindings: set, reader: str) -> Any:
        resolved = (
            self._resolve_read(value_id, bindings, reader)
            if self.enclosing_loop_states else None
        )
        return self.external_value(int(value_id)) if resolved is None else resolved

    def _callsite_argument(
        self, callsite_id: int, position: int, value_id: int,
    ) -> Any:
        """A call argument, read by the binding its operand read."""

        if self.lexical_read_scope is None or not self.enclosing_loop_states:
            return self.external_value(int(value_id))
        operand = self._book().page("call_argument_operand").latest(
            (self.lexical_read_scope, int(callsite_id), int(position))
        )
        bindings = (
            set() if operand is None
            else self._operand_bindings(((int(callsite_id), *operand),))
        )
        return self._read(
            value_id, bindings, f"callsite {callsite_id} argument {position}",
        )

    def _break_bound_initial(self, port_id: int, initial_id: int) -> Any:
        """A break-bound port's no-break value: its binding's pre-loop value.

        The port's own identity is the binding it continues
        (``loop_result_port_binding``).
        """

        if self.lexical_read_scope is None or not self.enclosing_loop_states:
            return self.external_value(int(initial_id))
        binding = self._book().page("loop_result_port_binding").latest(
            (self.lexical_read_scope, int(port_id))
        )
        if binding is None:
            raise ValueError(
                f"{self.function_name}: break-bound result port {port_id} "
                "has no loop_result_port_binding row"
            )
        return self._read(
            initial_id, {str(binding)}, f"result port {port_id}",
        )

    def _aliases_resolving_to(self, value_id: int) -> tuple[int, ...]:
        """The graph spellings of a region feed's planning resident.

        ``blocked[j] = 1.0`` followed by ``rhs[k] = blocked[k]`` in
        successive inner loops of one retained outer loop reads a LoopResult
        port in the graph and its resident span in region SSA.  Entering the
        loops redirects the live aliases to their updates; those temporal
        aliases cannot identify the graph operand whose lexical row we need.
        Use the same durable planning concordance that canonicalized the
        region operand, preserving its exact authored read position.
        """

        return tuple(
            int(alias)
            for alias in self.concorded_value_aliases
            if int(alias) != int(value_id)
            and self._concorded_resident(int(alias)) == int(value_id)
        )

    def _region_feed(self, region_index: int, value_id: int) -> Any:
        """A region feed, read by the bindings of the operands it serves."""

        book = self._book() if self.lexical_read_scope is not None else None
        split = None if book is None else book.page(
            "region_capture_binding"
        ).latest((
            str(self.tensor_shape_concordance_scope), int(region_index),
            int(value_id),
        ))
        if split is not None:
            # A formal the region split off one captured value: it is fed
            # that value as read by exactly one binding.
            source_id, binding = split
            if not self.enclosing_loop_states:
                return self.external_value(int(source_id))
            return self._read(
                int(source_id), {str(binding)},
                f"region {region_index} formal {value_id}",
            )
        item = None if book is None else book.page("item_operand").latest(
            (self.lexical_read_scope, int(value_id))
        )
        if item is not None:
            # A captured ``item()`` (planner page ``item_operand``): read its
            # operand by the binding at the item's operand position, in this
            # site's loop generation.  A rank-0 operand IS the item's value;
            # that merge is decided here, on the value read, and recorded.
            operand_id, role, ordinal = item
            source = (
                self._read(
                    int(operand_id),
                    self._operand_bindings(
                        ((int(value_id), role, int(ordinal)),)
                    ),
                    f"region {region_index} item {value_id}",
                )
                if self.enclosing_loop_states
                else self.external_value(int(operand_id))
            )
            if not tuple(source.shape or ()):
                # The rank-0 merge is one ``scalar_item_merge`` row (plan 80
                # B1.3), no longer a raw ``identity_transition`` revise.
                self._post_scalar_item_merge(int(value_id), source)
                return source
        if self.lexical_read_scope is None or not self.enclosing_loop_states:
            return self.external_value(int(value_id))
        feed_row = (
            str(self.tensor_shape_concordance_scope), int(region_index),
            int(value_id),
        )
        consumers = book.page(REGION_FEED_CONSUMER).latest(feed_row) or ()
        operand_page = book.page("consumer_operand")
        operands = []
        for consumer in consumers:
            positions = operand_page.latest(
                (self.lexical_read_scope, int(consumer), int(value_id))
            )
            if not positions:
                # The consumer's operand is a version that resolves to this
                # feed (an in-place store chain resolving to its arena): its
                # row is keyed by that version.
                positions = tuple(
                    position
                    for alias in self._aliases_resolving_to(int(value_id))
                    for position in (operand_page.latest(
                        (self.lexical_read_scope, int(consumer), alias)
                    ) or ())
                )
            if not positions:
                # No ``consumer_operand`` row: recorded before the fallback
                # (plan 80 B2.5), DERIVED(the feed's consumer cell).
                self._bind_unresolved(
                    int(value_id), REGION_FEED_NO_OPERAND_ROW,
                    book.latest_ref(REGION_FEED_CONSUMER, feed_row),
                    self._canonical_cell(consumer),
                )
                operands.append((int(consumer), None, 0))
                continue
            operands.extend(
                (int(consumer), role, ordinal) for role, ordinal in positions
            )
        return self._read(
            value_id, self._operand_bindings(operands),
            f"region {region_index}",
        )

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
            else self._region_feed(region_index, value_id)
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
        # REGION_CALL_RESULT: the region's signature cell (posted where the
        # signature was built) and the storage arguments it is fed.
        signature_cell = self._declared_cell(
            REGION_SIGNATURE.name, (self._scope(), int(region_index)),
        )
        aggregate = (
            self.fresh_value(
                dtype="ssa.aggregate", transform=REGION_CALL_RESULT,
                operands=(signature_cell, *storage_arguments),
            )
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
                index = self.constant_value(output_index, aggregate)
                attributes = {
                    "region_index": region_index,
                    "aggregate_index": output_index,
                    "source_output_id": output_id,
                }
                if int(output_id) in self.preserved_region_output_ids:
                    address = self.fresh_value(
                        dtype="ptr", transform=ADDRESS,
                        operands=(aggregate, index),
                    )
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
                        transform=LOAD, operands=(address,),
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

    def new_block(self, stem: str, *, also: tuple = ()) -> BasicBlock:
        """A new block.  ``also`` names cells beyond the owning control
        block's that the block is made from (a tick's case literal)."""

        count = self.block_counts.get(stem, 0)
        self.block_counts[stem] = count + 1
        name = stem if count == 0 else f"{stem}.{count}"
        block = BasicBlock(name)
        self.blocks[name] = block
        self._post_ssa_block(name, stem, also=also)
        return block

    def _post_ssa_block(
        self, label: str, stem: str, *, also: tuple = (),
    ) -> Ref:
        """``ssa_block`` row ``(scope, function, label)`` (plan 100, 2.6),
        CONCORD.  ENTRY / FUNCTION_EXIT belong to the function: DERIVED
        from the shell ``control_program`` cell (when posted) and the
        function root.  Any other block is DERIVED from the ``control_block``
        cell of the control block being lowered; when that block has no row
        of its own the fact is ``Unresolved(SSA_BLOCK_OWNER_UNROUTED)``
        reading the nearest enclosing control cell (else the function's)."""

        from .concordance_declarations import (
            CONTROL_PROGRAM, SHELL, SSA_BLOCK, SSA_BLOCK_OWNER_UNROUTED,
            SSABlockKind as _Kind,
        )

        # The stem is the declared kind's value: an undeclared stem raises.
        kind = _Kind(stem)
        book = self._book()
        program = (
            None if self.lexical_read_scope is None
            else book.latest_ref(
                CONTROL_PROGRAM, (tuple(self.lexical_read_scope), SHELL),
            )
        )
        function_cells = tuple(
            cell for cell in (program, self._function_root()) if cell is not None
        )
        fact: Any = kind
        if kind in {_Kind.ENTRY, _Kind.FUNCTION_EXIT}:
            sources = function_cells
        elif self._control_cells and self._control_cells[-1] is not None:
            sources = (self._control_cells[-1],)
        else:
            enclosing = next(
                (cell for cell in reversed(self._control_cells) if cell is not None),
                None,
            )
            sources = (enclosing,) if enclosing is not None else function_cells
            fact = Unresolved(SSA_BLOCK_OWNER_UNROUTED, sources)
        sources = tuple(dict.fromkeys((
            *sources, *(cell for cell in also if isinstance(cell, Ref)),
        )))
        return book.post(
            SSA_BLOCK, (self._scope(), str(self.function_name), str(label)),
            fact, stage=self._stage(), provenance=Derived(sources),
            mode=Mode.CONCORD,
        )

    def function_exit_block(self) -> BasicBlock:
        """The one block every source ``return`` branches to (created lazily).

        It is registered when first needed but only ever filled by `finish`,
        which merges the return edges per output slot and emits the Ret.
        """

        if self._function_exit is None:
            self._function_exit = self.new_block("function_exit")
        return self._function_exit

    def _value_dominates_block(
        self, value: SSAValue, current_name: str,
    ) -> bool:
        """Whether this exact SSA object is resident on a block's entry."""

        if any(argument is value for argument in self.arguments):
            return True
        producer_blocks = {
            name
            for name, block in self.blocks.items()
            if any(instruction.res is value for instruction in block.instrs)
        }
        if not producer_blocks:
            return False
        if current_name in producer_blocks:
            return True
        block_names = tuple(self.blocks)
        entry_name = block_names[0]
        predecessors = {name: set() for name in block_names}
        for name, block in self.blocks.items():
            for successor in block.successors:
                if successor in predecessors:
                    predecessors[successor].add(name)
        # Only blocks reachable from the entry take part.  An unreachable
        # block (the continuation minted after an unconditional break/return
        # inside an arm) has no predecessors; letting it into the fixpoint
        # collapsed every dominator set downstream of the merge it feeds to
        # the block itself, so a value defined before an inner loop was
        # judged non-dominating at that loop's exit.
        reachable = {entry_name}
        frontier = [entry_name]
        while frontier:
            name = frontier.pop()
            for successor in self.blocks[name].successors:
                if successor in predecessors and successor not in reachable:
                    reachable.add(successor)
                    frontier.append(successor)
        dominators = {
            name: ({name} if name == entry_name else set(block_names))
            for name in block_names
        }
        changed = True
        while changed:
            changed = False
            for name in block_names:
                if name == entry_name or name not in reachable:
                    continue
                incoming = {
                    parent for parent in predecessors[name]
                    if parent in reachable
                }
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
        # A carried "updated" slot is one SSAValue object written twice by
        # design: seeded in the preheader, rewritten in the body (see
        # ``lower_loop``).  Whichever of those writes dominates the current
        # block is the write the edge observes, so the slot is resident here
        # when ANY producer block dominates.  Requiring exactly one producer
        # rejected every carried slot at a break site, and the exit edge
        # silently carried the header Phi (the value from BEFORE this
        # iteration's update) instead.
        return any(
            producer in dominators[current_name]
            for producer in producer_blocks
        )

    def _value_dominates_current_edge(self, value: SSAValue) -> bool:
        """Whether this exact SSA object is resident on the current edge."""

        return self._value_dominates_block(value, self.current.name)

    def _is_stand_in(self, graph_id: int) -> bool:
        """Whether ``graph_id`` is bound only to the stand-in ``external_value``
        minted from absence (``Unresolved(NO_PRODUCER_AT_USE)`` on its binding
        row): no region has produced the id here."""

        fact = self._book().page(CONTROL_VALUE_BINDING).latest(
            (self._scope(), int(graph_id))
        )
        return (
            isinstance(fact, Unresolved)
            and fact.reason is NO_PRODUCER_AT_USE
        )

    def _phi_initial_binding(
        self, spelled_id: int, resident: SSAValue, phi_result: SSAValue,
        *, stage: Any,
    ) -> int:
        """Post ``phi_initial_binding`` for one Phi and return the resident id
        THE ROW states, which is what the Phi's ``initial_value_id`` carries.

        The graph speaks of the version that stood before a merge or a loop
        by its planning spelling (``spelled_id``: a graph id, a planning
        alias, a literal's graph node); the lowering holds that version as
        the SSA value ``resident``, whose id is not always the spelling (a
        planning alias of a region output, a minted literal, a restored
        carried slot).  An ``initial_value_id`` that kept the spelling named
        an id its function does not define.  DERIVED(the resident's and the
        Phi result's ``ssa_value`` cells).
        """

        from .concordance_declarations import PHI_INITIAL_BINDING

        book = self._book()
        row = (self._scope(), int(phi_result.id))
        book.post(
            PHI_INITIAL_BINDING, row, (int(spelled_id), int(resident.id)),
            stage=stage, provenance=Derived(self._cells(resident, phi_result)),
            mode=Mode.REVISE,
        )
        return int(book.page(PHI_INITIAL_BINDING).latest(row)[1])

    def _complete_current_join_value(
        self,
        candidate: SSAValue,
        incumbent: SSAValue,
        *,
        initial_value_id: int,
        initial_value: SSAValue,
        updated_value_id: int,
        site_node_id: int | None,
    ) -> SSAValue:
        """Make a partially defined carried value resident at this join.

        A predicated ``continue`` is represented after its enclosing arms have
        merged.  The value recorded at the source site can therefore dominate
        only the arm on which the continue predicate is true.  Complete that
        value with the loop-header incumbent on every other incoming edge.
        This is the ordinary SSA fixed point for the binding: an authored
        update wins where it is available; an equal-priority missing update
        retains the incumbent.
        """

        if self._value_dominates_current_edge(candidate):
            return candidate

        block_names = tuple(self.blocks)
        entry_name = block_names[0]
        predecessors = {name: [] for name in block_names}
        for name, block in self.blocks.items():
            for successor in block.successors:
                if successor in predecessors:
                    predecessors[successor].append(name)
        reachable = {entry_name}
        frontier = [entry_name]
        while frontier:
            name = frontier.pop()
            for successor in self.blocks[name].successors:
                if successor in predecessors and successor not in reachable:
                    reachable.add(successor)
                    frontier.append(successor)
        incoming_blocks = tuple(
            name for name in predecessors[self.current.name]
            if name in reachable
        )
        if not incoming_blocks:
            return incumbent
        incoming_values = tuple(
            candidate
            if self._value_dominates_block(candidate, predecessor)
            else incumbent
            for predecessor in incoming_blocks
        )
        if all(value is incoming_values[0] for value in incoming_values[1:]):
            return incoming_values[0]

        completed = self.fresh_value(
            dtype=str(candidate.dtype or incumbent.dtype or "unknown"),
            shape=tuple(candidate.shape or incumbent.shape),
            transform=PHI_LOOP_HEADER, operands=tuple(incoming_values),
        )
        self.emit(
            Handler.Phi,
            list(incoming_values),
            completed,
            attributes={
                "incoming_blocks": incoming_blocks,
                "binding": "loop_continue_carried",
                "initial_value_id": self._phi_initial_binding(
                    initial_value_id, initial_value, completed,
                    stage=CONTROL_SSA_LOOP,
                ),
                "initial_spelled_value_id": int(initial_value_id),
                "updated_value_id": int(updated_value_id),
                "site_node_id": site_node_id,
                "tie_policy": "incumbent",
            },
        )
        # Phis describe block entry and must precede any ordinary operation
        # already emitted into the merge block.
        emitted = self.current.instrs.pop()
        phi_end = next(
            (
                index for index, instruction in enumerate(self.current.instrs)
                if str(instruction.op).casefold() != "phi"
            ),
            len(self.current.instrs),
        )
        self.current.instrs.insert(phi_end, emitted)
        return completed

    def _complete_loop_latch_carried(
        self,
        carried: list[tuple[int, int, SSAValue, SSAValue, SSAValue]],
        carried_phis: dict[int, Instr],
        carried_updates: dict[int, SSAValue],
        continue_edges: list[tuple[BasicBlock, tuple[SSAValue, ...]]],
    ) -> None:
        """Resolve every reachable latch edge to one carried value per name."""

        block_names = tuple(self.blocks)
        entry_name = block_names[0]
        predecessors = {name: [] for name in block_names}
        for name, block in self.blocks.items():
            for successor in block.successors:
                if successor in predecessors:
                    predecessors[successor].append(name)
        reachable = {entry_name}
        frontier = [entry_name]
        while frontier:
            name = frontier.pop()
            for successor in self.blocks[name].successors:
                if successor in predecessors and successor not in reachable:
                    reachable.add(successor)
                    frontier.append(successor)
        incoming_blocks = tuple(
            name for name in predecessors[self.current.name]
            if name in reachable
        )
        continue_by_block = {
            edge.name: values for edge, values in continue_edges
            if edge.name in incoming_blocks
        }
        for index, (
            updated_id, initial_id, initial_value, reserved, incumbent
        ) in enumerate(carried):
            candidate = carried_updates[index]
            incoming_values = tuple(
                continue_by_block[predecessor][index]
                if predecessor in continue_by_block
                else (
                    candidate
                    if self._value_dominates_block(candidate, predecessor)
                    else incumbent
                )
                for predecessor in incoming_blocks
            )
            if not incoming_values:
                completed = incumbent
            elif all(
                value is incoming_values[0] for value in incoming_values[1:]
            ):
                completed = incoming_values[0]
            else:
                completed = self.fresh_value(
                    dtype=str(candidate.dtype or incumbent.dtype or "unknown"),
                    shape=tuple(candidate.shape or incumbent.shape),
                    transform=PHI_LOOP_HEADER, operands=tuple(incoming_values),
                )
                self.emit(
                    Handler.Phi,
                    list(incoming_values),
                    completed,
                    attributes={
                        "incoming_blocks": incoming_blocks,
                        "binding": "loop_latch_carried",
                        "initial_value_id": self._phi_initial_binding(
                            initial_id, initial_value, completed,
                            stage=CONTROL_SSA_LOOP,
                        ),
                        "initial_spelled_value_id": int(initial_id),
                        "updated_value_id": int(updated_id),
                        "tie_policy": "incumbent",
                    },
                )
            if os.environ.get("TURING_DEBUG_LOOP_ALIAS_PUBLICATION"):
                print(
                    "DEBUG-LOOP-LATCH-CHOICE "
                    f"fn={self.function_name} "
                    f"updated={int(updated_id)} initial={int(initial_id)} "
                    f"candidate={int(candidate.id)} "
                    f"incumbent={int(incumbent.id)} "
                    f"incoming_blocks={incoming_blocks!r} "
                    f"incoming={tuple(int(value.id) for value in incoming_values)!r} "
                    f"completed={int(completed.id)}",
                    file=sys.stderr,
                    flush=True,
                )
            carried_phis[index].args[1] = completed
            carried_updates[index] = completed
            # LOOP_LATCH: the body's binding of the update is the cause.
            self._bind(
                int(updated_id), completed, BindingKind.LOOP_LATCH,
                candidate, incumbent,
            )
            source_loop_node_id = carried_phis[index].attributes.get(
                "source_loop_node_id"
            )
            if (
                source_loop_node_id is not None
                and int(completed.id) != int(reserved.id)
            ):
                from .identity_concordance import rebind_loop_scope_inner

                rebind_loop_scope_inner(
                    self.function_name,
                    source_loop_node_id,
                    int(reserved.id),
                    int(completed.id),
                    "complete_loop_latch_carried",
                )
                completed.accounting = {
                    **dict(completed.accounting or {}),
                    "loop_scope_inner_transition": (
                        int(reserved.id), int(completed.id),
                        "complete_loop_latch_carried",
                    ),
                }

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
                extent = self.fresh_value(
                    dtype="int64", transform=LOAD,
                    operands=(length_cell, iterable_id),
                )
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
                pool_length = self.external_value(pool.length_value_id)
                length_address = self.fresh_value(
                    dtype="ptr", transform=ADDRESS,
                    operands=(pool_length, handle),
                )
                extent = self.fresh_value(
                    dtype="int", transform=LOAD, operands=(length_address,),
                )
                self.emit(
                    Handler.GetElementPtr,
                    [pool_length, handle],
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
                length_address = self.fresh_value(
                    dtype="ptr", transform=ADDRESS,
                    operands=(child_lengths, handle),
                )
                extent = self.fresh_value(
                    dtype="int", transform=LOAD, operands=(length_address,),
                )
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
            extent = self.fresh_value(
                dtype="int", transform=CONTROL_EXPRESSION, operands=(source,),
            )
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
            # In ``for i in range(n): for j in range(i, n): ...``, the
            # declared u_control_i initially names a formal, then the outer
            # loop binds its graph id to its induction Phi. Returning the
            # cached declaration froze the inner start and leaked that local
            # as an unnamed function input. The declaration names the id;
            # the current binding owns its value, as for arithmetic bounds.
            return self.external_value(int(uniform.id))
        local = self.local_control_values.get(spelling)
        if local is not None:
            return local
        try:
            literal = int(spelling, 10)
        except ValueError:
            lowered = self._control_arithmetic_value(spelling)
            if lowered is not None:
                return lowered
            value = self.fresh_value(dtype="int", transform=LOAD)
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
        value = self.fresh_value(dtype="int", transform=CONTROL_CONST)
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
                    else "float64",
                    transform=CONTROL_CONST,
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
                uniform = self.uniform_values.get(node.id)
                return (
                    self.external_value(int(uniform.id))
                    if uniform is not None else
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
                    transform=CONTROL_EXPRESSION, operands=(operand,),
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
                    transform=CONTROL_EXPRESSION, operands=(left, right),
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

    def _control_leaf(self, expression: ControlExpression) -> SSAValue:
        """A ``value`` leaf, read by the binding its operand read.

        The leaf carries its operand position (``read``); the binding is
        the book's ``lexical_read_binding`` fact.  Outside loops, or for a
        leaf with no recorded position, the value id is the answer.
        """

        value_id = int(expression.value_id)
        if (
            expression.read is None
            or self.lexical_read_scope is None
            or not self.enclosing_loop_states
        ):
            return self.external_value(value_id)
        return self._read(
            value_id,
            self._operand_bindings((tuple(expression.read),)),
            f"control expression leaf {value_id}",
        )

    def lower_control_expression(
        self,
        expression: ControlExpression,
        *,
        result_override: SSAValue | None = None,
    ) -> SSAValue:
        if expression.op == "value":
            return self._control_leaf(expression)
        if expression.op == "optional_presence":
            if expression.value_id is None:
                raise ValueError("optional presence requires a value id")
            value = self.external_value(
                int(expression.value_id), dtype="bool"
            )
            value.accounting = {
                **dict(value.accounting or {}),
                **({
                    "program_abi_record": expression.program_abi_record,
                } if expression.program_abi_record is not None else {}),
                **({
                    "program_abi_parameter": expression.program_abi_parameter,
                } if expression.program_abi_parameter is not None else {}),
                **({
                    "program_abi_field": expression.program_abi_field,
                } if expression.program_abi_field is not None else {}),
                "program_abi_storage": "scalar",
                "program_abi_optional_presence": True,
                "program_abi_optional_present_when": True,
                "physical_dtype": "bool",
                "physical_dtype_provenance": (
                    "control_ir_optional_presence"
                ),
                "physical_dtype_tie_policy": "incumbent",
            }
            return value
        if expression.op == "const":
            result = result_override or self.fresh_value(
                dtype="bool" if isinstance(expression.literal, bool) else None,
                transform=CONTROL_CONST, operands=(expression.value_id,),
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
            length = self.fresh_value(
                dtype="int", transform=LOAD,
                operands=(length_address, self._sequence_contract_cell(sequence_id)),
            )
            self.emit(
                Handler.Load,
                [length_address],
                length,
                attributes={
                    "binding": "ssa_sequence_length",
                    "sequence_value_id": sequence_id,
                },
            )
            zero = self.constant_value(0, length)
            result = result_override or self.fresh_value(
                dtype="bool", transform=CONTROL_PREDICATE, operands=(length, zero),
            )
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
                if expression.value_id is not None:
                    source_id = int(expression.value_id)
                    # A rank-0 operand IS the item's value: the merge is one
                    # ``scalar_item_merge`` row and the item id's binding.
                    self._post_scalar_item_merge(source_id, source)
                    self._bind(
                        source_id, source, BindingKind.CONTROL_EXPRESSION,
                        self._book().latest_ref(
                            SCALAR_ITEM_MERGE, (self._scope(), source_id),
                        ),
                    )
                    self.control_identity_receipts.append((
                        source_id, int(source.id), "scalar_item_identity",
                    ))
                return source
            index = self.constant_value(0, source)
            address = self.fresh_value(
                dtype="ptr", transform=ADDRESS, operands=(source, index),
            )
            self.emit(
                Handler.GetElementPtr, [source, index], address,
                attributes={"binding": "control_scalar_item"},
            )
            result = result_override or self.fresh_value(
                dtype=source.dtype, transform=LOAD, operands=(address,),
            )
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
            # Both native scalar emitters implement the registry's intrinsic
            # spelling. A Python math call must not survive this boundary.
            "isfinite": Handler.IsFinite,
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
            ),
            transform=CONTROL_EXPRESSION,
            operands=(*operand_values, expression.value_id),
        )
        self.emit(handler, operand_values, result, attributes={
            "binding": "control_expression",
            "source_value_id": expression.value_id,
        })
        if expression.value_id is not None:
            # CONTROL_EXPRESSION: the operand bindings are the cause.
            self._bind(
                int(expression.value_id), result,
                BindingKind.CONTROL_EXPRESSION, *operand_values,
            )
        return result

    def _post_scalar_item_merge(self, item_id: int, source: SSAValue) -> Ref:
        """``scalar_item_merge`` row ``(function scope, item)`` -> the
        merged source's ``ssa_value`` cell, DERIVED(the ``item_operand``
        cell when the planner declared it, the operand's binding cell)."""

        book = self._book()
        row = (self._scope(), int(item_id))
        source_cell = self._value_cell(source)
        latest = book.latest_ref(SCALAR_ITEM_MERGE, row)
        if latest is not None and (
            book.pages[SCALAR_ITEM_MERGE.name].cells.get((row, latest.column))
            == source_cell
        ):
            return latest
        sources = self._cells(
            self._declared_cell(
                "item_operand", (self.lexical_read_scope, int(item_id)),
            ) if self.lexical_read_scope is not None else None,
            self._binding_cell(int(source.id)), source_cell,
            self._canonical_cell(item_id),
        )
        return book.post(
            SCALAR_ITEM_MERGE, row, source_cell,
            stage=self.active_stage,
            provenance=Derived(sources),
            mode=Mode.CONCORD,
        )

    def lower(self, block: ControlBlock, *, path: str = "root") -> None:
        if os.environ.get("TURING_DEBUG_CONTROL_OVERLAY"):
            print(
                f"DEBUG-LOWER fn={self.function_name} path={path} "
                f"block={type(block).__name__}"
                + (
                    f" action={block.action} source_action={block.source_action}"
                    f" return_value_ids={block.return_value_ids!r}"
                    if isinstance(block, LoopControlBlock) else ""
                )
                + (
                    f" lines={block.lines!r}"
                    if isinstance(block, StatementBlock) else ""
                ),
                file=sys.stderr,
            )
        previous = self._evolution_source
        if self.evolution is not None:
            self._evolution_source = self.evolution.component_for_artifact(block)
        # A SequenceBlock is a container with no row (flattened for
        # placement); its children are owned by their own rows.
        owned = not isinstance(block, SequenceBlock)
        if owned:
            from .control_source import control_block_cell

            self._control_cells.append(control_block_cell(
                self._book(), self.lexical_read_scope, block,
            ))
        try:
            self._lower(block, path=path)
        finally:
            self._evolution_source = previous
            if owned:
                self._control_cells.pop()

    def _emit_resource_cleanup(self, action: str, path: str) -> None:
        for scope, loop_depth in reversed(self.resource_scopes):
            if action != "return" and loop_depth < len(self.loop_targets):
                continue  # This exit stays inside a scope surrounding the loop.
            for index, operation in enumerate(scope.cleanup):
                self.lower(operation, path=f"{path}.cleanup[{scope.source_scope_id}:{index}]")

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
        if isinstance(block, ResourceScopeBlock):
            self.resource_scopes.append((block, len(self.loop_targets)))
            try:
                self.lower(block.body, path=f"{path}.resource_body")
            finally:
                self.resource_scopes.pop()
            for index, operation in enumerate(block.cleanup):
                self.lower(operation, path=f"{path}.resource_cleanup[{index}]")
            return
        if isinstance(block, LoopControlBlock) and block.action == "return":
            # A source ``return`` below the function's top level: capture
            # this return's own slot values on the edge and branch to the
            # function exit, where every return edge is merged per slot
            # (`finish`). Loop-carried Phis at any enclosing loop exit do
            # not receive this edge -- the post-loop code does not run on
            # a return path -- which is exactly the semantics of ``return``.
            if os.environ.get("TURING_DEBUG_CONTROL_OVERLAY"):
                for slot in block.return_value_ids:
                    if slot is None:
                        continue
                    chain = []
                    current = int(slot)
                    while current in self.value_aliases and len(chain) < 8:
                        current = int(self.value_aliases[current])
                        chain.append(current)
                    resident = self.external_values.get(int(slot))
                    print(
                        f"DEBUG-RETURN-SLOT fn={self.function_name} slot={slot} "
                        f"alias_chain={chain!r} "
                        f"external_values[slot]={None if resident is None else int(resident.id)} "
                        f"external_values[chain_end]={None if self.external_values.get(current) is None else int(self.external_values[current].id)}",
                        file=sys.stderr,
                    )
            edge_values = tuple(
                None if slot is None else self.external_value(int(slot))
                for slot in block.return_value_ids
            )
            exit_block = self.function_exit_block()
            if block.predicate_value_id is None and (
                block.predicate_expression is None
            ):
                self._emit_resource_cleanup("return", path)
                self.function_return_edges.append((self.current, edge_values))
                self.branch(exit_block)
                self.current.instrs[-1].attributes["source_control"] = "return"
                self.current.instrs[-1].attributes["return_source_value_ids"] = tuple(block.return_value_ids)
                if block.return_site_cell is not None:
                    # The edge names its own return SITE (the construct's
                    # cell), not only the values it carries.
                    self.current.instrs[-1].attributes["return_site_cell"] = block.return_site_cell
                self.current = self.new_block("unreachable_return_control")
            else:
                fallthrough = self.new_block("return_control_next")
                predicate = (
                    self.lower_control_expression(block.predicate_expression)
                    if block.predicate_expression is not None
                    else self.external_value(
                        block.predicate_value_id, dtype="bool"
                    )
                )
                # The predicated edge leaves from a block of its own so the
                # exit Phi's incoming block names exactly one return site.
                returning = self.new_block("return_edge")
                self.conditional_branch(
                    predicate,
                    returning if block.expect_true else fallthrough,
                    fallthrough if block.expect_true else returning,
                )
                self.current.instrs[-1].attributes["source_control"] = "return"
                self.current = returning
                self._emit_resource_cleanup("return", path)
                self.function_return_edges.append((self.current, edge_values))
                self.branch(exit_block)
                self.current.instrs[-1].attributes["return_source_value_ids"] = tuple(block.return_value_ids)
                if block.return_site_cell is not None:
                    self.current.instrs[-1].attributes["return_site_cell"] = block.return_site_cell
                self.current = fallthrough
            return
        if isinstance(block, LoopControlBlock):
            if self.resource_scopes and (
                block.predicate_value_id is not None or block.predicate_expression is not None
            ):
                predicate = (
                    self.lower_control_expression(block.predicate_expression)
                    if block.predicate_expression is not None else
                    self.external_value(block.predicate_value_id, dtype="bool")
                )
                leaving = self.new_block("resource_exit")
                fallthrough = self.new_block("resource_exit_next")
                self.conditional_branch(predicate,
                                        leaving if block.expect_true else fallthrough,
                                        fallthrough if block.expect_true else leaving)
                self.current = leaving
                self.lower(replace(block, predicate_value_id=None, predicate_expression=None), path=path)
                self.current = fallthrough
                return
            if not self.loop_targets:
                raise ValueError(
                    f"{block.action} appears outside a loop "
                    f"(site {block.site_node_id}, {self.function_name}, {path})"
                )
            latch, exit_block = self.loop_targets[-1]
            target = exit_block if block.action == "break" else latch
            self._emit_resource_cleanup(block.action, path)
            if self.loop_exit_contexts and block.site_node_id is not None:
                self.loop_exit_contexts[-1]["sites_seen"].add(
                    int(block.site_node_id)
                )
            if block.action in {"break", "continue"} and self.loop_exit_contexts:
                context = self.loop_exit_contexts[-1]
                # The reducer recorded, per site, the value every rebound
                # name holds THERE (keyed by its pre-loop identity).  That is
                # the value the exit edge carries; the older
                # "updated-if-it-dominates-else-current" rule remains only
                # for names the site table does not mention.
                site_values = {
                    int(initial_id): int(value_id)
                    for initial_id, value_id in block.site_values
                }

                def site_value(
                    initial_id: int,
                    updated_id: int,
                    incumbent: SSAValue,
                    initial_value: SSAValue,
                ) -> SSAValue | None:
                    value_id = site_values.get(int(initial_id))
                    if value_id is None:
                        return None
                    value = self.external_values.get(int(value_id))
                    if value is None and int(value_id) in self.constant_value_ids:
                        # A folded literal (``flag = True; break``): the
                        # control function materializes it in its entry.
                        value = self.external_value(int(value_id))
                    if value is not None and self._value_dominates_current_edge(value):
                        return value
                    if value is not None and block.action == "continue":
                        return self._complete_current_join_value(
                            value,
                            incumbent,
                            initial_value_id=int(initial_id),
                            initial_value=initial_value,
                            updated_value_id=int(updated_id),
                            site_node_id=block.site_node_id,
                        )
                    if value is None or block.action == "break":
                        if os.environ.get("TURING_DEBUG_BREAK_EDGE"):
                            print(
                                f"DEBUG-BREAK-EDGE {self.function_name} site="
                                f"{block.site_node_id} initial={initial_id} "
                                f"value_id={value_id} object={value!r} "
                                f"current={self.current.name}",
                                file=sys.stderr, flush=True,
                            )
                            for name, emitted in self.blocks.items():
                                for instruction in emitted.instrs:
                                    print(
                                        f"    {name}: {instruction.op} "
                                        f"{[int(a.id) for a in instruction.args]} -> "
                                        f"{None if instruction.res is None else int(instruction.res.id)}"
                                        f"{' <== SITE VALUE OBJECT' if value is not None and instruction.res is value else ''}",
                                        file=sys.stderr, flush=True,
                                    )
                        self.shortfalls.append(SSALoweringShortfall(
                            "control", "break-edge-value", path,
                            f"break site {block.site_node_id} carries value "
                            f"{value_id} for pre-loop identity {initial_id} "
                            "but that value does not dominate the exit edge",
                        ))
                        return None
                    return value

                carried_values = []
                for updated_id, initial_id, current, initial_value in (
                    context["carried"]
                ):
                    at_site = site_value(
                        int(initial_id), int(updated_id), current,
                        initial_value,
                    )
                    if at_site is not None:
                        carried_values.append(at_site)
                        continue
                    candidate = self.external_values.get(
                        int(updated_id),
                        self.external_values.get(int(initial_id), current),
                    )
                    if self._value_dominates_current_edge(candidate):
                        carried_values.append(candidate)
                    elif block.action == "continue":
                        carried_values.append(self._complete_current_join_value(
                            candidate,
                            current,
                            initial_value_id=int(initial_id),
                            initial_value=initial_value,
                            updated_value_id=int(updated_id),
                            site_node_id=block.site_node_id,
                        ))
                    else:
                        carried_values.append(current)
                if block.action == "break":
                    bound_values = []
                    for _port_id, initial_id in context["break_bound"]:
                        incumbent = self.external_value(int(initial_id))
                        at_site = site_value(
                            int(initial_id), int(initial_id), incumbent,
                            incumbent,
                        )
                        bound_values.append(
                            at_site if at_site is not None else incumbent
                        )
                    context["break_edges"].append((
                        self.current, tuple(carried_values), tuple(bound_values)
                    ))
                else:
                    context["continue_edges"].append((
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
        if isinstance(block, DispatchBlock):
            arguments = [self.external_value(value_id)
                         for value_id in block.argument_value_ids]
            arguments.extend(self.external_value(value_id)
                             for _name, value_id in block.keyword_argument_value_ids)
            result = None if block.result_value_id is None else SSAValue(
                int(block.result_value_id), dtype=str(block.result_dtype)
            )
            self.emit(Handler.Dispatch, arguments, result, attributes={
                "dispatch_operation": str(block.operation),
                "source_callsite_id": int(block.callsite_id),
                "keyword_names": tuple(name for name, _ in block.keyword_argument_value_ids),
                "deployment_owner": "dispatcher",
                "required_capability": "communicating_tasks",
                "effectful": True,
                "ordered_effect": True,
                "extraction_identity": "turing.dispatch." + str(block.operation),
            })
            if result is not None:
                # CALL_RESULT: the dispatch's argument cells are the cause.
                self._bind(
                    int(result.id), result, BindingKind.CALL_RESULT, *arguments,
                )
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
            # CALL_RESULT: the external call's argument cells are the cause.
            self._bind(
                int(block.result_value_id), result, BindingKind.CALL_RESULT,
                *arguments,
            )
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
        if isinstance(block, ScalarFieldWriteBlock):
            value = self.lower_control_expression(block.value_expression)
            if block.field_value_id is not None:
                destination = self.external_value(
                    block.field_value_id, dtype=block.dtype,
                )
                self.emit(Handler.Store, [value, destination], attributes={
                    "source_effect_node_id": block.effect_node_id,
                    # The write's own field-state cell: the version published
                    # below begins at THIS effect (record-return selection
                    # starts its intervening-write scan after it).
                    "field_state_cell": block.field_state_cell,
                    "binding": "scalar_record_field_assignment",
                })
            # The write IS one ``reducer_field_state`` cell (its WRITTEN
            # revision).  The emitted value is that cell's SSA version: post
            # it on ``ssa_field_version`` at (function scope, cell) so the
            # conditional merge and the record-return selection read the
            # version at the cell the reducer's merge named, not under a
            # graph id.  Local returned records have no destination cell
            # here; their record-return materialization consumes this same
            # version.
            version_cell = None
            if block.field_state_cell is not None:
                version_cell = self._publish_field_version(
                    block.field_state_cell, value, (block.field_state_cell,),
                )
            # FIELD_WRITE: the SetAttr id resolves to the written value; the
            # version cell lane C posts is its cause.
            self._bind(
                int(block.effect_node_id), value, BindingKind.FIELD_WRITE,
                version_cell, block.field_state_cell,
            )
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
        nested_tensor: bool = False,
        nested_value_dtype: str | None = None,
        element_dtype: str | None = None,
        column_dtypes: tuple[str | None, ...] = (),
        column_shapes: tuple[tuple[int, ...], ...] = (),
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
            nested_tensor = bool(nested_tensor or resolved.nested_tensor)
            nested_value_dtype = nested_value_dtype or resolved.nested_value_dtype
        key_columns = (
            tuple(range(max(1, int(column_count) - 1)))
            if policy == "unique" else ()
        )
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
        proposed_column_dtypes = tuple(
            _canonicalize_column(
                first_dtype if column == 0 else (
                    column_dtypes[column]
                    if column < len(column_dtypes) else None
                ),
                column,
            ) or "unknown"
            for column in range(int(column_count))
        )
        proposed_column_shapes = tuple(
            tuple(map(int, column_shapes[column]))
            if column < len(column_shapes) else ()
            for column in range(int(column_count))
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
            concorded_column_dtypes = concord_sequence_row_dtypes(
                self.function_name,
                {value_id: proposed_column_dtypes},
                source=f"{location}: descriptor reuse",
            )
            if tuple(existing.column_dtypes) != concorded_column_dtypes:
                existing = replace(
                    existing, column_dtypes=concorded_column_dtypes
                )
                self.sequence_descriptors[value_id] = existing
                for value, dtype in zip(
                    self.sequence_storage_values[value_id],
                    concorded_column_dtypes,
                ):
                    if str(value.dtype or "unknown") in {
                        "", "None", "unknown",
                    }:
                        value.dtype = str(dtype)
            incumbent_shapes = tuple(existing.column_shapes) or tuple(
                () for _column in existing.column_value_ids
            )
            merged_shapes = []
            for column, (incumbent, proposed) in enumerate(zip(
                incumbent_shapes, proposed_column_shapes, strict=True,
            )):
                if incumbent and proposed and incumbent != proposed:
                    self.shortfalls.append(SSALoweringShortfall(
                        "ssa-sequence", "conflicting-row-shape", location,
                        f"sequence value {value_id} column {column} receives "
                        f"both {incumbent!r} and {proposed!r}",
                    ))
                    return None
                merged_shapes.append(incumbent or proposed)
            if tuple(merged_shapes) != incumbent_shapes:
                existing = replace(
                    existing, column_shapes=tuple(merged_shapes),
                )
                self.sequence_descriptors[value_id] = existing
            return existing
        concorded_column_dtypes = concord_sequence_row_dtypes(
            self.function_name,
            {value_id: proposed_column_dtypes},
            source=f"{location}: descriptor construction",
        )
        if concorded_column_dtypes:
            first_dtype = concorded_column_dtypes[0]
        data = self.external_value(value_id, dtype=first_dtype)
        # Claim the physical arena at its creation boundary.  Waiting for the
        # final storage-declaration sweep is too late: call/result identity
        # reconciliation may otherwise treat this provisional formal as a
        # replaceable source value and substitute a loop-local scalar.  The
        # storage claim is incumbent provenance; later equal-priority source
        # aliases must retain it.
        data.accounting = {
            **dict(data.accounting or {}),
            "compiler_frame_storage": str(self.function_name),
            "compiler_frame_sequence_id": value_id,
            "compiler_frame_member": 0,
            "sequence_arena": True,
            "transformation_priority": "compiler_storage_identity",
            "transformation_tie_policy": "incumbent",
            "sequence_row_shape": proposed_column_shapes[0],
        }
        if first_dtype is not None and element_dtype is not None:
            # An explicit element contract (literal bytes, joined outer-row
            # handles, fill materialization) is stronger than the provisional
            # float dtype assigned when a structural constant first entered
            # the control value map.
            data.dtype = str(first_dtype)
        # Every storage cell of the descriptor descends from the data arena
        # (the authored sequence's value) and its contract cell.
        descriptor_operands = (data, self._sequence_contract_cell(int(value_id)))
        extra_columns = tuple(
            self.fresh_value(dtype=_canonicalize_column(
                (
                    "int64"
                    if nested_table and index == int(column_count) - 2
                    else concorded_column_dtypes[index + 1]
                ),
                index + 1,
            ), transform=DESCRIPTOR_CELL, operands=descriptor_operands)
            for index in range(max(0, int(column_count) - 1))
        )
        for index, column in enumerate(extra_columns, start=1):
            column.accounting = {
                **dict(column.accounting or {}),
                "compiler_frame_storage": str(self.function_name),
                "compiler_frame_sequence_id": value_id,
                "compiler_frame_member": int(index),
                "sequence_arena": True,
                "sequence_row_shape": proposed_column_shapes[index],
                "transformation_priority": "compiler_storage_identity",
                "transformation_tie_policy": "incumbent",
            }
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
        def storage_cell(dtype: str, shape: tuple = ()) -> SSAValue:
            return self.fresh_value(
                dtype=dtype, shape=shape, transform=DESCRIPTOR_CELL,
                operands=descriptor_operands,
            )

        length_address = storage_cell("int64", (1,))
        capacity = storage_cell("int64")
        requires_status = bool(writable) or policy == "unique"
        status_address = (
            storage_cell("int", (1,))
            if requires_status else None
        )
        self.arguments.extend((length_address, capacity))
        if status_address is not None:
            self.arguments.append(status_address)
        live_flags = (
            storage_cell("bool") if retains_deleted_rows else None
        )
        if live_flags is not None:
            self.arguments.append(live_flags)
        child_table_pool = None
        child_pool_values: tuple[SSAValue, ...] = ()
        if nested_table:
            child_keys = (
                None
                if nested_tensor
                else storage_cell("unknown")
            )
            child_values = storage_cell(nested_value_dtype or "unknown")
            child_lengths = storage_cell("int")
            child_capacity = storage_cell("int")
            child_stride = storage_cell("int")
            child_shapes = (
                storage_cell("int32") if nested_tensor else None
            )
            child_ranks = (
                storage_cell("int32") if nested_tensor else None
            )
            child_shape_stride = (
                storage_cell("int64") if nested_tensor else None
            )
            child_status = storage_cell("int")
            child_live = storage_cell("bool")
            child_columns = (
                (child_values,) if child_keys is None
                else (child_keys, child_values)
            )
            child_pool_values = (
                *child_columns, child_lengths, child_capacity,
                child_stride,
                *((child_shapes, child_ranks, child_shape_stride)
                  if nested_tensor else ()),
                child_status, child_live,
            )
            self.arguments.extend(child_pool_values)
            # The pool cells are bound under their own (minted) ids: the
            # sequence contract is their cause.
            for pool_value in child_pool_values:
                self._bind(
                    int(pool_value.id), pool_value,
                    BindingKind.SEQUENCE_RESULT, *descriptor_operands,
                )
            child_table_pool = SSAChildTablePoolDescriptor(
                handle_column=1,
                column_value_ids=tuple(
                    int(column.id) for column in child_columns
                ),
                length_value_id=int(child_lengths.id),
                capacity_value_id=int(child_capacity.id),
                row_stride_value_id=int(child_stride.id),
                shape_value_id=(
                    None if child_shapes is None else int(child_shapes.id)
                ),
                rank_value_id=(
                    None if child_ranks is None else int(child_ranks.id)
                ),
                shape_stride_value_id=(
                    None
                    if child_shape_stride is None
                    else int(child_shape_stride.id)
                ),
                status_value_id=int(child_status.id),
                live_flags_value_id=int(child_live.id),
                column_dtypes=(
                    (nested_value_dtype or "unknown",)
                    if nested_tensor
                    else ("unknown", nested_value_dtype or "unknown")
                ),
                key_columns=(() if nested_tensor else (0,)),
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
            column_shapes=proposed_column_shapes,
            key_columns=key_columns,
            live_flags_value_id=(
                None if live_flags is None else int(live_flags.id)
            ),
            capacity_policy=SSASequenceCapacityPolicy.FIXED,
            writable=bool(writable),
            child_table_pool=child_table_pool,
        )
        # NOTE: a key column here is routinely left with the provisional
        # dtype a structural constant carries (float64, or "unknown"), and
        # no column contract reaches this point to correct it -- see
        # ``identity_concordance``'s ``key-column-not-integral`` finding,
        # which reports it for the whole module.  It is NOT raised as an
        # ``SSALoweringShortfall`` here on purpose: a shortfall is fatal at
        # emission, the defect is present for every keyed store in the
        # tree, and making it fatal halts every build before the root cause
        # (the missing contract) can be threaded down to this point.  The
        # audit carries it until then.
        self.sequence_descriptors[value_id] = descriptor
        self.sequence_storage_values[value_id] = (
            data, *extra_columns, length_address, capacity,
            *((live_flags,) if live_flags is not None else ()),
        )
        if status_address is not None:
            self.sequence_status_values[value_id] = status_address
        return descriptor

    def _sequence_status_address(self, status_arena: SSAValue) -> SSAValue:
        zero = self.constant_value(0, status_arena)
        address = self.fresh_value(
            dtype="ptr", transform=ADDRESS, operands=(status_arena, zero),
        )
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
        length = self.fresh_value(
            dtype="int64", transform=LOAD, operands=(length_address,),
        )
        self.emit(Handler.Load, [length_address], length, attributes=attributes)
        if query.operation == "maximum":
            value_dtype = str(storage[0].dtype or "unknown")

            def ordered_maximum(
                incumbent: SSAValue,
                candidate: SSAValue,
            ) -> SSAValue:
                """Select candidate only for strict greater-than.

                This is Python's ordered ``max`` rule: equality and unordered
                floating-point comparisons retain the incumbent.
                """

                selected = self.new_block("sequence_maximum_selected")
                retained = self.new_block("sequence_maximum_retained")
                complete = self.new_block("sequence_maximum_merge")
                greater = self.fresh_value(
                    dtype="bool", transform=CONTROL_PREDICATE,
                    operands=(candidate, incumbent),
                )
                self.emit(
                    Handler.Gt,
                    [candidate, incumbent],
                    greater,
                    attributes=attributes,
                )
                self.conditional_branch(greater, selected, retained)
                self.current = selected
                self.branch(complete)
                self.current = retained
                self.branch(complete)
                self.current = complete
                merged = self.fresh_value(
                    dtype=value_dtype, transform=PHI_CONDITIONAL,
                    operands=(candidate, incumbent, greater),
                )
                self.emit(
                    Handler.Phi,
                    [candidate, incumbent],
                    merged,
                    attributes={
                        **attributes,
                        "incoming_blocks": (selected.name, retained.name),
                        "binding": "ssa_sequence_ordered_maximum",
                    },
                )
                return merged

            prefix = tuple(
                self.external_value(int(value_id), dtype=value_dtype)
                for value_id in query.reduction_prefix_value_ids
            )
            accumulator = prefix[0]
            for candidate in prefix[1:]:
                accumulator = ordered_maximum(accumulator, candidate)

            preheader = self.current
            header = self.new_block("sequence_maximum_header")
            body = self.new_block("sequence_maximum_body")
            selected = self.new_block("sequence_maximum_loop_selected")
            retained = self.new_block("sequence_maximum_loop_retained")
            latch = self.new_block("sequence_maximum_latch")
            exit_block = self.new_block("sequence_maximum_exit")
            zero = self.constant_value(0, length)
            one = self.constant_value(1, length)
            next_index = self.fresh_value(
                dtype="int64", transform=CONTROL_EXPRESSION, operands=(length,),
            )
            next_accumulator = self.fresh_value(
                dtype=value_dtype, transform=PHI_CONDITIONAL,
                operands=(accumulator, storage[0]),
            )
            self.branch(header)

            self.current = header
            index = self.fresh_value(
                dtype="int64", transform=PHI_LOOP_HEADER,
                operands=(zero, next_index),
            )
            self.emit(
                Handler.Phi,
                [zero, next_index],
                index,
                attributes={
                    **attributes,
                    "incoming_blocks": (preheader.name, latch.name),
                    "binding": "ssa_sequence_maximum_index",
                },
            )
            current_accumulator = self.fresh_value(
                dtype=value_dtype, transform=PHI_LOOP_HEADER,
                operands=(accumulator, next_accumulator),
            )
            self.emit(
                Handler.Phi,
                [accumulator, next_accumulator],
                current_accumulator,
                attributes={
                    **attributes,
                    "incoming_blocks": (preheader.name, latch.name),
                    "binding": "ssa_sequence_maximum_accumulator",
                },
            )
            has_item = self.fresh_value(
                dtype="bool", transform=CONTROL_PREDICATE,
                operands=(index, length),
            )
            self.emit(Handler.Lt, [index, length], has_item, attributes=attributes)
            self.conditional_branch(has_item, body, exit_block)

            self.current = body
            address = self.fresh_value(
                dtype="ptr", transform=ADDRESS, operands=(storage[0], index),
            )
            self.emit(
                Handler.GetElementPtr,
                [storage[0], index],
                address,
                attributes=attributes,
            )
            candidate = self.fresh_value(
                dtype=value_dtype, transform=LOAD, operands=(address,),
            )
            self.emit(Handler.Load, [address], candidate, attributes=attributes)
            greater = self.fresh_value(
                dtype="bool", transform=CONTROL_PREDICATE,
                operands=(candidate, current_accumulator),
            )
            self.emit(
                Handler.Gt,
                [candidate, current_accumulator],
                greater,
                attributes=attributes,
            )
            self.conditional_branch(greater, selected, retained)
            self.current = selected
            self.branch(latch)
            self.current = retained
            self.branch(latch)

            self.current = latch
            self.emit(
                Handler.Phi,
                [candidate, current_accumulator],
                next_accumulator,
                attributes={
                    **attributes,
                    "incoming_blocks": (selected.name, retained.name),
                    "binding": "ssa_sequence_ordered_maximum",
                },
            )
            self.emit(
                Handler.Add,
                [index, one],
                next_index,
                attributes=attributes,
            )
            self.branch(header)

            self.current = exit_block
            accumulator = current_accumulator
            for value_id in query.reduction_suffix_value_ids:
                accumulator = ordered_maximum(
                    accumulator,
                    self.external_value(int(value_id), dtype=value_dtype),
                )
            result = self.produced_value(
                int(query.result_value_id),
                dtype=value_dtype,
                claim_provisional_definition=True,
                cause=(accumulator,),
            )
            self.emit(
                Handler.Cast,
                [accumulator],
                result,
                attributes={**attributes, "target_dtype": value_dtype},
            )
            self._bind_sequence_query_result(query, result, accumulator)
            return
        if query.operation in {"length", "truth"}:
            result = self.produced_value(
                int(query.result_value_id),
                dtype="bool" if query.operation == "truth" else "int64",
                claim_provisional_definition=True,
                cause=(length,),
            )
            self.emit(
                Handler.Gt if query.operation == "truth" else Handler.Cast,
                [length, self.constant_value(0, length)] if query.operation == "truth" else [length],
                result,
                attributes={**attributes, "target_dtype": result.dtype},
            )
            self._bind_sequence_query_result(query, result, length)
            return

        zero = self.constant_value(0, length)
        nonempty = self.fresh_value(
            dtype="bool", transform=CONTROL_PREDICATE, operands=(length, zero),
        )
        self.emit(Handler.Gt, [length, zero], nonempty, attributes=attributes)
        selected = self.new_block("sequence_query_selected")
        defaulted = self.new_block("sequence_query_defaulted")
        complete = self.new_block("sequence_query_merge")
        self.conditional_branch(nonempty, selected, defaulted)

        self.current = selected
        address = self.fresh_value(
            dtype="ptr", transform=ADDRESS, operands=(storage[0], zero),
        )
        self.emit(
            Handler.GetElementPtr,
            [storage[0], zero],
            address,
            attributes=attributes,
        )
        selected_value = self.fresh_value(
            dtype=("int" if query.row_handle else storage[0].dtype),
            transform=LOAD, operands=(address,),
        )
        self.emit(Handler.Load, [address], selected_value, attributes=attributes)
        self.branch(complete)

        self.current = defaulted
        default_value = (
            self.constant_value(-1, storage[0])
            if query.row_handle
            else self.external_value(int(query.default_value_id))
        )
        self.branch(complete)

        self.current = complete
        result = self.produced_value(
            int(query.result_value_id),
            dtype=str(selected_value.dtype or default_value.dtype or "unknown"),
            claim_provisional_definition=True,
            cause=(selected_value, default_value, nonempty),
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
        self._bind_sequence_query_result(
            query, result, selected_value, default_value,
        )

    def _bind_sequence_query_result(
        self, query: Any, result: SSAValue, *cells: Any,
    ) -> None:
        """Bind a sequence query's result id and its alias ids to ``result``
        (SEQUENCE_RESULT, from the sequence's contract cell and ``cells``)."""

        contract_cell = self._sequence_contract_cell(int(query.sequence_value_id))
        self._bind(
            int(query.result_value_id), result, BindingKind.SEQUENCE_RESULT,
            contract_cell, *cells,
        )
        for alias_id in query.result_alias_ids:
            self._bind(
                int(alias_id), result, BindingKind.SEQUENCE_RESULT,
                contract_cell, *cells,
            )

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
            length = self.fresh_value(
                dtype="int64", transform=LOAD, operands=(length_address,),
            )
            self.emit(
                Handler.Load, [length_address], length,
                attributes=attributes,
            )
            nonempty = self.fresh_value(
                dtype="bool", transform=CONTROL_PREDICATE, operands=(length,),
            )
            self.emit(
                Handler.Gt,
                [length, self.constant_value(0, length)],
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
            new_length = self.fresh_value(
                dtype="int64", transform=CONTROL_EXPRESSION, operands=(length,),
            )
            self.emit(
                Handler.Sub,
                [length, self.constant_value(1, length)],
                new_length,
                attributes=attributes,
            )
            address = self.fresh_value(
                dtype="ptr", transform=ADDRESS, operands=(storage[0], new_length),
            )
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
                cause=(address,),
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
            self._bind(
                int(mutation.effect_node_id), result,
                BindingKind.SEQUENCE_RESULT, address,
            )
            return

        if operation == "remove" and not mutation.argument_kind.startswith(
            "mapping_"
        ):
            expected_columns = len(destination.column_value_ids)
            if len(mutation.argument_value_ids) != expected_columns:
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence", operation, location,
                    "resident sequence remove requires one explicit value "
                    f"per row column; expected {expected_columns}, received "
                    f"{len(mutation.argument_value_ids)}",
                ))
                return
            queries = tuple(
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
            storage = self.sequence_storage_values[
                int(destination.sequence_id)
            ]
            length_address = storage[len(destination.column_value_ids)]
            attributes = {
                "binding": "ssa_sequence_remove",
                "sequence_id": int(destination.sequence_id),
                "source_effect_node_id": int(mutation.effect_node_id),
            }
            length = self.fresh_value(
                dtype="int64", transform=LOAD, operands=(length_address,),
            )
            self.emit(Handler.Load, [length_address], length, attributes=attributes)
            zero = self.constant_value(0, length)
            one = self.constant_value(1, length)
            scan_header = self.new_block("sequence_remove_scan")
            scan_body = self.new_block("sequence_remove_compare")
            scan_latch = self.new_block("sequence_remove_next")
            shift_header = self.new_block("sequence_remove_shift")
            shift_body = self.new_block("sequence_remove_shift_row")
            shift_latch = self.new_block("sequence_remove_shift_next")
            store_length = self.new_block("sequence_remove_store_length")
            absent = self.new_block("sequence_remove_absent")
            complete = self.new_block("sequence_remove_complete")
            scan_entry = self.current
            self.branch(scan_header)

            self.current = scan_header
            next_scan = self.fresh_value(
                dtype="int64", transform=CONTROL_EXPRESSION, operands=(length,),
            )
            scan_index = self.fresh_value(
                dtype="int64", transform=PHI_LOOP_HEADER,
                operands=(zero, next_scan),
            )
            self.emit(
                Handler.Phi, [zero, next_scan], scan_index,
                attributes={
                    **attributes,
                    "incoming_blocks": (scan_entry.name, scan_latch.name),
                },
            )
            scanning = self.fresh_value(
                dtype="bool", transform=CONTROL_PREDICATE,
                operands=(scan_index, length),
            )
            self.emit(Handler.Lt, [scan_index, length], scanning,
                      attributes=attributes)
            self.conditional_branch(scanning, scan_body, absent)

            self.current = scan_body
            matches = None
            for column, query in enumerate(queries):
                address = self.fresh_value(
                    dtype="ptr", transform=ADDRESS,
                    operands=(storage[column], scan_index),
                )
                existing = self.fresh_value(
                    dtype=str(destination.column_dtypes[column]),
                    transform=LOAD, operands=(address,),
                )
                equal = self.fresh_value(
                    dtype="bool", transform=CONTROL_PREDICATE,
                    operands=(existing, query),
                )
                self.emit(
                    Handler.GetElementPtr,
                    [storage[column], scan_index], address,
                    attributes=attributes,
                )
                self.emit(Handler.Load, [address], existing,
                          attributes=attributes)
                self.emit(Handler.Eq, [existing, query], equal,
                          attributes=attributes)
                if matches is None:
                    matches = equal
                else:
                    combined = self.fresh_value(
                        dtype="bool", transform=CONTROL_PREDICATE,
                        operands=(matches, equal),
                    )
                    self.emit(Handler.LAnd, [matches, equal], combined,
                              attributes=attributes)
                    matches = combined
            self.conditional_branch(matches, shift_header, scan_latch)

            self.current = scan_latch
            self.emit(Handler.Add, [scan_index, one], next_scan,
                      attributes=attributes)
            self.branch(scan_header)

            new_length = self.fresh_value(
                dtype="int64", transform=CONTROL_EXPRESSION,
                operands=(length, one),
            )
            self.current = shift_header
            self.emit(Handler.Sub, [length, one], new_length,
                      attributes=attributes)
            next_shift = self.fresh_value(
                dtype="int64", transform=CONTROL_EXPRESSION,
                operands=(scan_index, one),
            )
            shift_index = self.fresh_value(
                dtype="int64", transform=PHI_LOOP_HEADER,
                operands=(scan_index, next_shift),
            )
            self.emit(
                Handler.Phi, [scan_index, next_shift], shift_index,
                attributes={
                    **attributes,
                    "incoming_blocks": (scan_body.name, shift_latch.name),
                },
            )
            shifting = self.fresh_value(
                dtype="bool", transform=CONTROL_PREDICATE,
                operands=(shift_index, new_length),
            )
            self.emit(Handler.Lt, [shift_index, new_length], shifting,
                      attributes=attributes)
            self.conditional_branch(shifting, shift_body, store_length)

            self.current = shift_body
            source_index = self.fresh_value(
                dtype="int64", transform=CONTROL_EXPRESSION,
                operands=(shift_index, one),
            )
            self.emit(Handler.Add, [shift_index, one], source_index,
                      attributes=attributes)
            for column, dtype in enumerate(destination.column_dtypes):
                source_address = self.fresh_value(
                    dtype="ptr", transform=ADDRESS,
                    operands=(storage[column], source_index),
                )
                destination_address = self.fresh_value(
                    dtype="ptr", transform=ADDRESS,
                    operands=(storage[column], shift_index),
                )
                value = self.fresh_value(
                    dtype=str(dtype), transform=LOAD, operands=(source_address,),
                )
                self.emit(
                    Handler.GetElementPtr,
                    [storage[column], source_index], source_address,
                    attributes=attributes,
                )
                self.emit(Handler.Load, [source_address], value,
                          attributes=attributes)
                self.emit(
                    Handler.GetElementPtr,
                    [storage[column], shift_index], destination_address,
                    attributes=attributes,
                )
                self.emit(Handler.Store, [value, destination_address],
                          attributes=attributes)
            self.branch(shift_latch)

            self.current = shift_latch
            self.emit(Handler.Add, [shift_index, one], next_shift,
                      attributes=attributes)
            self.branch(shift_header)

            self.current = store_length
            self.emit(Handler.Store, [new_length, length_address],
                      attributes=attributes)
            self.branch(complete)

            self.current = absent
            self.emit(
                Handler.Call, [],
                attributes={
                    "callee": "turing_validation_error",
                    "error_code": 72,
                    **attributes,
                },
            )
            self.branch(complete)
            self.current = complete
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

                lookup_name = self._sequence_helper_name(
                    int(destination.sequence_id), "setdefault_lookup",
                )
                lookup = lower_table_lookup(
                    destination,
                    function_name=lookup_name,
                    first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
                )
                self._register_sequence_lowering(lookup)
                lookup_handle = self.fresh_value(
                    dtype="int", transform=TABLE_LOOKUP,
                    operands=(
                        self._sequence_contract_cell(int(destination.sequence_id)),
                        *mutation.argument_value_ids[:key_count],
                    ),
                )
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
                status_cell = self._sequence_status_address(
                    self.sequence_status_values[int(destination.sequence_id)]
                )
                status = self.fresh_value(
                    dtype="int", transform=LOAD, operands=(status_cell,),
                )
                self.emit(
                    Handler.Load,
                    [status_cell],
                    status,
                    attributes={"binding": "ssa_mapping_setdefault_status"},
                )
                found = self.fresh_value(
                    dtype="bool", transform=CONTROL_PREDICATE, operands=(status,),
                )
                self.emit(
                    Handler.Gt, [status, self.constant_value(0, status)], found,
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
                outer_length = self.fresh_value(
                    dtype="int64", transform=LOAD, operands=(outer_length_address,),
                )
                self.emit(
                    Handler.Load, [outer_length_address], outer_length,
                    attributes={"binding": "ssa_mapping_setdefault_handle"},
                )
                new_handle = self.fresh_value(
                    dtype="int", transform=CONTROL_CAST, operands=(outer_length,),
                )
                self.emit(
                    Handler.Cast, [outer_length], new_handle,
                    attributes={"target_dtype": "int"},
                )
                self._bind(
                    int(new_handle.id), new_handle, BindingKind.SEQUENCE_RESULT,
                    outer_length,
                )
                pool_length = self.external_value(int(pool.length_value_id))
                child_length_address = self.fresh_value(
                    dtype="ptr", transform=ADDRESS,
                    operands=(pool_length, new_handle),
                )
                self.emit(
                    Handler.GetElementPtr,
                    [pool_length, new_handle],
                    child_length_address,
                    attributes={"binding": "ssa_child_table_length"},
                )
                self.emit(
                    Handler.Store,
                    [self.constant_value(0, child_length_address), child_length_address],
                    attributes={"binding": "ssa_child_table_initialize"},
                )
                if pool.status_value_id is not None:
                    pool_status = self.external_value(int(pool.status_value_id))
                    child_status_address = self.fresh_value(
                        dtype="ptr", transform=ADDRESS,
                        operands=(pool_status, new_handle),
                    )
                    self.emit(
                        Handler.GetElementPtr,
                        [pool_status, new_handle],
                        child_status_address,
                        attributes={"binding": "ssa_child_table_status"},
                    )
                    self.emit(
                        Handler.Store,
                        [self.constant_value(0, child_status_address), child_status_address],
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
                    cause=(lookup_handle, new_handle, found),
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
                self._bind(
                    int(mutation.effect_node_id), result,
                    BindingKind.SEQUENCE_RESULT, lookup_handle, new_handle,
                )
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
            lower_sequence_replace,
        )

        call_arguments: tuple[SSAValue, ...]
        deferred_record_row: tuple[int, str, int] | None = None
        deferred_record_slots: tuple[tuple[int, int, str, int], ...] = ()
        indexed_record_row_source: tuple[int, int, str] | None = None
        if operation in {"append", "add"}:
            expected_columns = len(destination.column_value_ids)
            record_slots = self.sequence_row_record_slots.get(
                int(destination.sequence_id), ()
            )
            if record_slots and len(
                mutation.argument_value_ids
            ) != expected_columns:
                # A record named at one column of the authored row stands
                # for its member fields.  Its physical layout is proven by
                # native-call linking, which expands the single semantic
                # operand in place; here the row is checked to be exactly
                # the authored width once every record column is expanded.
                argument_count = len(mutation.argument_value_ids)
                applicable = tuple(
                    (position, identity, width)
                    for position, identity, width in record_slots
                    if position < argument_count
                )
                expanded_width = argument_count + sum(
                    width - 1 for _position, _identity, width in applicable
                )
                if applicable and expanded_width == expected_columns:
                    deferred_record_slots = tuple(
                        (
                            int(position),
                            int(mutation.argument_value_ids[position]),
                            str(identity),
                            int(width),
                        )
                        for position, identity, width in applicable
                    )
            if (
                not deferred_record_slots
                and len(mutation.argument_value_ids) != expected_columns
            ):
                record_identity = self.sequence_record_identities.get(
                    int(destination.sequence_id)
                )
                if (
                    record_identity is not None
                    and len(mutation.argument_value_ids) == 1
                    and expected_columns > 1
                ):
                    row_value_id = int(mutation.argument_value_ids[0])
                    row_sources = tuple(
                        (
                            int(iterable_id), int(target_id),
                            str(induction_name),
                        )
                        for iterable_id, target_id, induction_name, projection
                        in self.program.projected_iterable_bindings
                        if int(target_id) == row_value_id
                        and projection == "induction"
                    )
                    row_source = (
                        None if len(row_sources) != 1
                        else self.sequence_descriptors.get(
                            int(row_sources[0][0])
                        )
                    )
                    source_row_identity = (
                        None if row_source is None
                        else self.sequence_record_identities.get(
                            int(row_source.sequence_id)
                        )
                    )
                    if (
                        row_source is not None
                        and len(row_source.column_value_ids)
                        == int(expected_columns)
                        and (
                            source_row_identity is None
                            or str(source_row_identity) == str(record_identity)
                            or str(source_row_identity).rsplit(".", 1)[-1]
                            == str(record_identity).rsplit(".", 1)[-1]
                        )
                    ):
                        indexed_record_row_source = row_sources[0]
                    else:
                        deferred_record_row = (
                            row_value_id,
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
                count_name = self._sequence_helper_name(
                    int(destination.sequence_id), "append",
                )
                count_lowering = lower_sequence_append(
                    destination,
                    function_name=count_name,
                    first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
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
                flat_name = self._sequence_helper_name(
                    int(joined_flat_id),
                    (
                        "append_singleton"
                        if singleton_value_id is not None else "extend"
                    ),
                    (
                        int(source_id)
                        if singleton_value_id is not None
                        else int(source.sequence_id)
                    ),
                )
                flat_lowering = (
                    lower_sequence_append(
                        flat,
                        function_name=flat_name,
                        first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
                    )
                    if singleton_value_id is not None
                    else lower_sequence_extend(
                        flat,
                        source,
                        function_name=flat_name,
                        first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
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
                destination_storage = self.sequence_storage_values[
                    destination.sequence_id
                ]
                count_status = self.fresh_value(
                    dtype="int", transform=REGION_CALL_RESULT,
                    operands=tuple(destination_storage),
                )
                self.emit(
                    Handler.Call,
                    [
                        *destination_storage,
                        self.constant_value(0, *destination_storage),
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
                flat_status = self.fresh_value(
                    dtype="int", transform=REGION_CALL_RESULT,
                    operands=tuple(joined_call_values.values()),
                )
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
            function_name = self._sequence_helper_name(
                int(destination.sequence_id), str(operation),
            )
            lowering = (
                lower_sequence_append(
                    destination,
                    function_name=function_name,
                    first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
                )
                if operation == "append"
                else lower_sequence_add(
                    destination,
                    function_name=function_name,
                    first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
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
            if indexed_record_row_source is not None:
                source_id, _row_target_id, induction_name = (
                    indexed_record_row_source
                )
                source_descriptor = self.sequence_descriptors[int(source_id)]
                source_columns = self.sequence_storage_values[int(source_id)][
                    :len(source_descriptor.column_value_ids)
                ]
                row_index = mutation_values[0]
                indexed_values = []
                for column_index, (source_column, column_dtype) in enumerate(
                    zip(
                        source_columns,
                        source_descriptor.column_dtypes,
                        strict=True,
                    )
                ):
                    existing_targets = tuple(
                        int(target_id)
                        for (
                            iterable_id, target_id, projected_induction,
                            projection,
                        ) in self.program.projected_iterable_bindings
                        if int(iterable_id) == int(source_id)
                        and str(projected_induction) == str(induction_name)
                        and projection == int(column_index)
                        and int(target_id) in self.external_values
                    )
                    if len(existing_targets) == 1:
                        projected = self.external_values[existing_targets[0]]
                    else:
                        projected = self.indexed_load(
                            source_column,
                            row_index,
                            int(self.fresh_value(
                                dtype=str(source_column.dtype or "unknown"),
                                transform=ROW_COLUMN_PROJECTION,
                                operands=(source_column, row_index),
                            ).id),
                            attributes={
                                "binding": "record_sequence_row_column",
                                "record_identity": str(
                                    self.sequence_record_identities.get(
                                        int(destination.sequence_id), ""
                                    )
                                ),
                                "source_sequence_id": int(source_id),
                                "column_index": int(column_index),
                            },
                        )
                    if str(column_dtype) not in {"", "unknown", "None"}:
                        projected.dtype = str(column_dtype)
                    indexed_values.append(projected)
                mutation_values = tuple(indexed_values)
            if deferred_record_row is None and not deferred_record_slots:
                for mutation_value, element_dtype in zip(
                    mutation_values, destination.column_dtypes
                ):
                    if str(element_dtype) not in {"", "unknown", "None"}:
                        mutation_value.dtype = str(element_dtype)
            elif deferred_record_slots:
                # Scalar columns keep their contract; a record column's
                # dtypes are applied by the link-time expansion.
                slot_positions = {
                    position for position, *_rest in deferred_record_slots
                }
                column_index = 0
                for position, mutation_value in enumerate(mutation_values):
                    slot = next((
                        item for item in deferred_record_slots
                        if item[0] == position
                    ), None)
                    if slot is not None:
                        column_index += int(slot[3])
                        continue
                    if column_index < len(destination.column_dtypes):
                        element_dtype = destination.column_dtypes[column_index]
                        if str(element_dtype) not in {"", "unknown", "None"}:
                            mutation_value.dtype = str(element_dtype)
                    column_index += 1
                del slot_positions
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
                policy=mutation.policy if operation == "replace" else "duplicates",
                writable=False,
                location=location,
            )
            if source is None:
                return
            concorded_dtypes = concord_sequence_row_dtypes(
                self.function_name,
                {
                    int(destination.sequence_id): destination.column_dtypes,
                    int(source.sequence_id): source.column_dtypes,
                },
                source=f"{location}: {operation} equivalence",
            )
            if tuple(destination.column_dtypes) != concorded_dtypes:
                destination = replace(
                    destination, column_dtypes=concorded_dtypes
                )
                self.sequence_descriptors[
                    int(destination.sequence_id)
                ] = destination
            if tuple(source.column_dtypes) != concorded_dtypes:
                source = replace(source, column_dtypes=concorded_dtypes)
                self.sequence_descriptors[int(source.sequence_id)] = source
            for descriptor in (destination, source):
                for value, dtype in zip(
                    self.sequence_storage_values[
                        int(descriptor.sequence_id)
                    ],
                    concorded_dtypes,
                ):
                    if str(value.dtype or "unknown") in {
                        "", "None", "unknown",
                    }:
                        value.dtype = str(dtype)
            function_name = self._sequence_helper_name(
                int(destination.sequence_id), str(operation),
                int(source.sequence_id),
            )
            lowering = (lower_sequence_replace if operation == "replace" else lower_sequence_extend)(
                destination,
                source,
                function_name=function_name,
                first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
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
        status = self.fresh_value(
            dtype="int", transform=REGION_CALL_RESULT,
            operands=tuple(call_arguments),
        )
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
                    "ssa_indexed_record_row": indexed_record_row_source,
                } if indexed_record_row_source is not None else {}),
                **({
                    "ssa_deferred_record_slots": (
                        len(call_arguments) - len(mutation.argument_value_ids),
                        deferred_record_slots,
                    ),
                } if deferred_record_slots else {}),
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
            handle_value = self.external_value(handle_id, dtype="int")
            predicate = self.fresh_value(
                dtype="bool", transform=CONTROL_PREDICATE,
                operands=(handle_value,),
            )
            self.emit(
                Handler.Ge,
                [handle_value, self.constant_value(0, handle_value)],
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
            length = self.fresh_value(
                dtype="int64", transform=LOAD,
                operands=(length_cell, predicate_sequence_id),
            )
            predicate = self.fresh_value(
                dtype="bool", transform=CONTROL_PREDICATE, operands=(length,),
            )
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
                [length, self.constant_value(0, length)],
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
            predicate = self.fresh_value(
                dtype="bool", transform=CONTROL_PREDICATE,
                operands=(length, predicate_source_id),
            )
            self.emit(
                Handler.Gt,
                [length, self.constant_value(0, length)],
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
        # One ``carried_snapshot`` row per snapshotted id: the binding cell
        # current at branch entry, DERIVED from that cell and the
        # conditional construct's identity cell (plan 80 B2.3).
        conditional_cell = self._construct_cell(conditional.source_node_id)
        snapshot_cells = self._post_carried_snapshots(
            conditional_cell, carried_snapshots,
        )
        stage_before = self.active_stage
        self.active_stage = CONTROL_SSA_CONDITIONAL

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
        # An arm that leaves through its own edge (``continue``/``break``
        # placed in the arm) never reaches the merge: nothing it binds may
        # reach the continuation.  The snapshot above covers only merged
        # identities; a non-falling-through arm restores every binding.
        # (``if retries < 2: dt = dt * 0.5; continue`` -- the arm published
        # ``dt`` = ``dt * 0.5`` and the fall-through call read it.)
        values_before_body = dict(self.external_values)
        self.lower(conditional.body, path=f"{path}.body")
        body_falls_through = (
            not self.current.successors
            or merge_block.name in self.current.successors
        )
        # Name-carried arms (decision 7.2): an arm that did not rebind the
        # name takes the entered version; an arm whose version the builder
        # cannot find is refused with ``Unresolved(NAME_ARM_VERSION_MISSING)``.
        true_carried = {}
        carried_name_sources: dict[int, list[Any]] = {}
        for true_id, _false_id, initial_id, _merged_id in (
            conditional.carried_aliases
        ):
            arm_value, arm_source = self._carried_name_arm(
                true_id, initial_id, carried_snapshots[int(initial_id)],
                snapshot_cells.get(int(initial_id)), path=f"{path}.body",
            )
            true_carried[int(initial_id)] = arm_value
            carried_name_sources[int(initial_id)] = [arm_source]
        # Field-carried merges name their arms as ``reducer_field_state``
        # cells; the arm's value is the version posted AT that cell, never a
        # graph-id lookup that may miss and default to the snapshot.
        carried_field_cells = tuple(conditional.carried_field_cells)
        carried_field_cells += (None,) * (
            len(conditional.carried_aliases) - len(carried_field_cells)
        )
        carried_field_sources: dict[int, list[Any]] = {}
        for alias, cells in zip(conditional.carried_aliases, carried_field_cells):
            if cells is None:
                continue
            initial_id = int(alias[2])
            true_value, true_source = self._carried_field_arm(
                cells[0], cells[1], carried_snapshots[initial_id],
                path=f"{path}.body", arm_value_id=int(alias[0]),
            )
            true_carried[initial_id] = true_value
            carried_field_sources[initial_id] = [true_source]
        true_results = {
            int(result_id): self.external_value(int(true_id))
            for true_id, _false_id, result_id in conditional.result_aliases
        }
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
        if not body_falls_through:
            self._restore_view(values_before_body, conditional_cell)

        # Each arm starts with the same incumbent versions.  Retain the true
        # arm's exact values above, then restore these identities before
        # lowering the false arm so in-place SetAttr effects cannot leak across
        # the branch boundary.  RESTORED, from the snapshot cells.
        for snapshot_id, snapshot_value in carried_snapshots.items():
            if self.external_values.get(int(snapshot_id)) is not snapshot_value:
                self._bind(
                    int(snapshot_id), snapshot_value, BindingKind.RESTORED,
                    snapshot_cells.get(int(snapshot_id)), conditional_cell,
                )
        self.current = false_block
        values_before_orelse = dict(self.external_values)
        if conditional.orelse is not None:
            self.lower(conditional.orelse, path=f"{path}.orelse")
        if self.current.successors and (
            merge_block.name not in self.current.successors
        ):
            self._restore_view(values_before_orelse, conditional_cell)
        false_results = {
            int(result_id): self.external_value(int(false_id))
            for _true_id, false_id, result_id in conditional.result_aliases
        }
        false_carried = {}
        for _true_id, false_id, initial_id, _merged_id in (
            conditional.carried_aliases
        ):
            arm_value, arm_source = self._carried_name_arm(
                false_id, initial_id, carried_snapshots[int(initial_id)],
                snapshot_cells.get(int(initial_id)), path=f"{path}.orelse",
            )
            false_carried[int(initial_id)] = arm_value
            carried_name_sources.setdefault(int(initial_id), []).append(arm_source)
        for alias, cells in zip(conditional.carried_aliases, carried_field_cells):
            if cells is None:
                continue
            initial_id = int(alias[2])
            false_value, false_source = self._carried_field_arm(
                cells[1], cells[0], carried_snapshots[initial_id],
                path=f"{path}.orelse", arm_value_id=int(alias[1]),
            )
            false_carried[initial_id] = false_value
            carried_field_sources.setdefault(initial_id, []).append(false_source)
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
        for _true_id, _false_id, result_id in conditional.result_aliases:
            true_value = true_results[int(result_id)]
            false_value = false_results[int(result_id)]
            merged = self.produced_value(
                int(result_id), claim_provisional_definition=True,
                cause=(true_value, false_value, predicate),
            )
            if merged.dtype in {None, "unknown"}:
                merged.dtype = true_value.dtype or false_value.dtype
            if (
                int(result_id) not in self.region_value_meta
                and tuple(true_value.shape) == tuple(false_value.shape)
                and true_value.dtype == false_value.dtype
            ):
                merged.shape = tuple(true_value.shape)
                merged.device = true_value.device
            self._revise_value(merged, true_value, false_value)
            self.emit(
                Handler.Phi, [true_value, false_value], merged,
                attributes={
                    "incoming_blocks": (true_exit.name, false_exit.name),
                    "binding": "conditional_result",
                },
            )
        for (
            true_value_id, false_value_id, initial_value_id, merged_value_id,
        ), field_cells in zip(conditional.carried_aliases, carried_field_cells):
            initial = carried_snapshots[int(initial_value_id)]
            true_value = true_carried[int(initial_value_id)]
            false_value = false_carried[int(initial_value_id)]
            # The Phi's operands on the book: each arm's source cell (its
            # binding, or the snapshot cell when it did not write) and the
            # snapshot cell -- PHI_CONDITIONAL (plan 80 B2.3).
            phi_operands = (
                *carried_name_sources.get(int(initial_value_id), ()),
                snapshot_cells.get(int(initial_value_id)),
                true_value, false_value, predicate,
            )
            if int(merged_value_id) in {
                int(true_value.id), int(false_value.id),
            }:
                # Graph identities may reuse the selected arm's SetAttr id as
                # the enclosing join id.  They are the same source version,
                # but the join is a distinct SSA definition.  Allocate that
                # definition separately and retain the graph id as provenance
                # instead of emitting a self-referential Phi.
                merged = self.fresh_value(
                    dtype=initial.dtype, shape=initial.shape,
                    transform=PHI_CONDITIONAL, operands=phi_operands,
                )
                merged.accounting.update({
                    "source_value_id": int(merged_value_id),
                    "ssa_conditional_write_version": True,
                })
            else:
                merged = SSAValue(
                    int(merged_value_id),
                    dtype=initial.dtype,
                    shape=initial.shape,
                )
                # The join adopts the graph's merge id; its row derives from
                # the same operands the Phi is made from.
                self._value_cell(merged, *phi_operands)
            # The version that stood before the branch.  The planner spells it
            # ``initial_value_id``; when no region ever produced that spelling
            # (``NO_PRODUCER_AT_USE``: ``external_value`` handed back a stand-in
            # formal) the entered version is the arm value that was already
            # bound when the branch was entered -- the arm that did not
            # rebind the name (``_carried_name_arm``).  N=2 orbital dt system,
            # ``_apply_energy_sidechain``: alias (65, 55, 64, 66), 64 never
            # produced, 55 the first conditional's merge.
            entered = initial
            if self._is_stand_in(initial_value_id):
                candidates = [
                    arm_value for arm_id, arm_value in (
                        (true_value_id, true_value),
                        (false_value_id, false_value),
                    )
                    if values_before_body.get(int(arm_id)) is arm_value
                ]
                if len(candidates) == 1:
                    entered = candidates[0]
            self.emit(
                Handler.Phi,
                [true_value, false_value],
                merged,
                attributes={
                    "incoming_blocks": (true_exit.name, false_exit.name),
                    "binding": "conditional_carried",
                    "initial_value_id": self._phi_initial_binding(
                        initial_value_id, entered, merged,
                        stage=CONTROL_SSA_CONDITIONAL,
                    ),
                    "initial_spelled_value_id": int(initial_value_id),
                },
            )
            # The merge IS the reducer's MERGED cell: publish the Phi as that
            # cell's SSA version, derived from the two arm sources (each arm's
            # version cell, or the arm cell itself when it did not write) and
            # the merged cell.  An arm recorded ARM_VERSION_MISSING has no
            # source; the merge then posts nothing rather than post a
            # snapshot as an arm.
            if field_cells is not None and field_cells[2] is not None:
                sources = carried_field_sources.get(int(initial_value_id), [])
                if len(sources) == 2 and all(
                    isinstance(source, Ref) for source in sources
                ):
                    self._publish_field_version(
                        field_cells[2], merged,
                        (sources[0], sources[1], field_cells[2]),
                    )
            # Every source identity participating in this join denotes the
            # merged version in the lexical continuation.  A later region may
            # legitimately define one of the arm ids again; lowering that
            # region overwrites the entry as usual.  Until then, retaining a
            # branch-local arm value here makes a following conditional read
            # a value that does not dominate it (and silently skips the first
            # conditional on the opposite arm).
            protected_sources = frozenset().union(
                *self.protected_loop_alias_sources
            ) if self.protected_loop_alias_sources else frozenset()
            published = {int(merged_value_id): merged}
            if int(initial_value_id) not in protected_sources:
                published[int(initial_value_id)] = merged
            for arm_value_id in (true_value_id, false_value_id):
                if int(arm_value_id) not in protected_sources:
                    published[int(arm_value_id)] = merged
            if (
                os.environ.get("TURING_DEBUG_LOOP_ALIAS_PUBLICATION")
                and protected_sources
                and protected_sources.intersection({
                    int(true_value_id), int(false_value_id),
                    int(initial_value_id), int(merged_value_id),
                })
            ):
                print(
                    "DEBUG-LOOP-ALIAS-PUBLICATION "
                    f"fn={self.function_name} path={path} "
                    f"alias={(int(true_value_id), int(false_value_id), int(initial_value_id), int(merged_value_id))!r} "
                    f"protected={tuple(sorted(protected_sources))!r} "
                    f"published={tuple(sorted(published))!r} "
                    f"merged_ssa={int(merged.id)}",
                    file=sys.stderr,
                    flush=True,
                )
            # CONDITIONAL_MERGE: every published id derives from the Phi.
            self._rebind_view(
                published, BindingKind.CONDITIONAL_MERGE, merged, conditional_cell,
            )
        for _true_id, _false_id, initial_id, merged_id in (
            conditional.carried_sequence_aliases
        ):
            resident = self.external_value(int(initial_id))
            self._bind(
                int(merged_id), resident, BindingKind.CONDITIONAL_MERGE,
                self._binding_cell(int(initial_id)), conditional_cell,
            )
        self.active_stage = stage_before

    def _publish_loop_result_ports(
        self,
        loop: LoopBlock | WhileBlock,
        *,
        header: BasicBlock,
        exit_block: BasicBlock,
        carried: list[tuple[int, int, SSAValue, SSAValue, SSAValue]],
        break_edges: tuple[
            tuple[BasicBlock, tuple[SSAValue, ...], tuple[SSAValue, ...]], ...
        ],
        break_bound: tuple[tuple[int, int], ...] = (),
        break_bound_initials: Mapping[int, SSAValue] | None = None,
    ) -> None:
        """Define authored LoopResult ids with edge-correct exit Phis."""

        # A port continues one authored binding; the carried entry it exits
        # is the one the book attributes that binding to.  Value ids do not
        # decide: two bindings can share an updated id (``total = w``) or an
        # initial id (``second = value``).
        from .identity_concordance import current_identity_book

        book = current_identity_book()
        loop_node = getattr(loop, "source_loop_node_id", None)
        entry_bindings = [
            tuple(book.page("loop_carried_binding").latest((
                self.lexical_read_scope, int(loop_node),
                int(updated_id), int(initial_id),
            )) or ())
            if self.lexical_read_scope is not None and loop_node is not None
            else ()
            for updated_id, initial_id, *_rest in carried
        ]

        def carried_entry_of(port_id: int, initial_id: int, updated_id: int):
            if self.lexical_read_scope is None or loop_node is None:
                # No book facts exist for a program the reducer never saw.
                return next((
                    (index, entry[4]) for index, entry in enumerate(carried)
                    if int(entry[0]) == int(updated_id)
                    and int(entry[1]) == int(initial_id)
                ), None)
            binding = book.page("loop_result_port_binding").latest(
                (self.lexical_read_scope, int(port_id))
            )
            if binding is None:
                raise ValueError(
                    f"{self.function_name}: loop {loop_node} result port "
                    f"{port_id} has no loop_result_port_binding row"
                )
            owners = [
                index for index, bindings in enumerate(entry_bindings)
                if binding in bindings
            ]
            if len(owners) > 1:
                raise ValueError(
                    f"{self.function_name}: binding {binding!r} is carried by "
                    f"loop {loop_node} entries {owners!r}"
                )
            return (owners[0], carried[owners[0]][4]) if owners else None

        def port_cells(port_id: int) -> tuple:
            """The port's ``loop_result_port_binding`` and
            ``loop_carried_entry`` cells, when the book has them."""

            if self.lexical_read_scope is None or loop_node is None:
                return ()
            port_binding = book.latest_ref(
                LOOP_RESULT_PORT_BINDING, (self.lexical_read_scope, int(port_id)),
            )
            binding = book.page(LOOP_RESULT_PORT_BINDING).latest(
                (self.lexical_read_scope, int(port_id))
            )
            entry_cell = None if binding is None else book.latest_ref(
                LOOP_CARRIED_ENTRY,
                (str(self.tensor_shape_concordance_scope), int(loop_node), str(binding)),
            )
            return (port_binding, entry_cell)

        def publish_port(port_id: int, port: SSAValue, *cells: Any) -> None:
            """LOOP_RESULT_PORT binding plus the ``carried_port_value`` row:
            the exit Phi's ``ssa_value`` cell, DERIVED(the port's binding
            row, its carried entry, the Phi cell)."""

            sources = self._cells(*port_cells(port_id), port, *cells)
            row = (self._scope(), int(port_id))
            phi_cell = self._value_cell(port)
            existing = book.latest_ref(CARRIED_PORT_VALUE, row)
            if existing is None or (
                book.pages[CARRIED_PORT_VALUE.name].cells.get(
                    (row, existing.column)
                ) != phi_cell
            ):
                book.post(
                    CARRIED_PORT_VALUE, row, phi_cell,
                    stage=CONTROL_SSA_LOOP,
                    provenance=Derived(sources),
                    mode=Mode.REVISE,
                )
            self._bind(
                int(port_id), port, BindingKind.LOOP_RESULT_PORT,
                book.latest_ref(CARRIED_PORT_VALUE, row), *cells,
            )
            port_values[int(port_id)] = port
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
            carried_entry = carried_entry_of(
                int(port_id), int(initial_id), int(updated_id),
            )
            if carried_entry is None:
                continue
            carried_index, normal_value = carried_entry
            incoming_blocks = [header.name]
            incoming_values = [normal_value]
            for predecessor, edge_values, _bound_values in break_edges:
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
                cause=tuple(incoming_values),
            )
            existing_definition = next((
                (block.name, instruction_index)
                for block in self.blocks.values()
                for instruction_index, instruction in enumerate(block.instrs)
                if instruction.res is not None
                and int(instruction.res.id) == int(port.id)
            ), None)
            if existing_definition is not None:
                # A carried body's latest lexical version can share the
                # LoopResult's graph spelling. It is already a definition and
                # cannot also become the exit Phi: doing so moves its only
                # numeric-id definition into the future of body uses. Version
                # the exit and retain the authored port id in the exact map
                # below. Only a producerless provisional formal is claimed in
                # place by produced_value above.
                previous_id = int(port.id)
                previous_port = port
                port = self.fresh_value(
                    dtype=str(normal_value.dtype or port.dtype or "unknown"),
                    shape=tuple(normal_value.shape or port.shape),
                    transform=LOOP_RESULT_VERSION,
                    operands=(previous_port, normal_value),
                )
                port.accounting.update({
                    "source_value_id": int(port_id),
                    "ssa_loop_result_version": True,
                    "versioned_from_value_id": previous_id,
                    "versioned_from_definition": existing_definition,
                    "tie_policy": "incumbent_body_definition",
                })
                self._bind(
                    int(port_id), port, BindingKind.LOOP_RESULT_PORT,
                    previous_port, normal_value,
                )
            port.shape = tuple(normal_value.shape)
            port.device = normal_value.device
            self._revise_value(port, normal_value)
            self.emit(
                Handler.Phi,
                incoming_values,
                port,
                attributes={
                    "incoming_blocks": tuple(incoming_blocks),
                    "binding": "loop_result_port",
                    "initial_value_id": self._phi_initial_binding(
                        initial_id, carried[carried_index][2], port,
                        stage=CONTROL_SSA_LOOP,
                    ),
                    "initial_spelled_value_id": int(initial_id),
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
                publish_port(int(equivalent_port_id), port, *incoming_values)
        # A break-bound name: zero-trip / fall-through exits keep the
        # pre-loop value; each break edge carries the value its site bound.
        for bound_index, (port_id, initial_id) in enumerate(break_bound):
            normal_value = (
                (break_bound_initials or {}).get(int(initial_id))
                or self.external_value(int(initial_id))
            )
            incoming_blocks = [header.name]
            incoming_values = [normal_value]
            for predecessor, _edge_values, bound_values in break_edges:
                incoming_blocks.append(predecessor.name)
                incoming_values.append(bound_values[bound_index])
            port = self.produced_value(
                int(port_id),
                dtype=str(normal_value.dtype or "unknown"),
                claim_provisional_definition=True,
                cause=tuple(incoming_values),
            )
            self.emit(
                Handler.Phi,
                incoming_values,
                port,
                attributes={
                    "incoming_blocks": tuple(incoming_blocks),
                    "binding": "loop_result_port",
                    "result_kind": "break_bound",
                    "initial_value_id": self._phi_initial_binding(
                        initial_id, normal_value, port,
                        stage=CONTROL_SSA_LOOP,
                    ),
                    "initial_spelled_value_id": int(initial_id),
                    "updated_value_id": int(initial_id),
                },
            )
            publish_port(int(port_id), port, *incoming_values)

    def _bind_loop_result_ports_inside_body(
        self, loop: LoopBlock | WhileBlock,
    ) -> tuple[tuple[int, bool, int | None], ...]:
        """Resolve an exit spelling to its live carried version in the loop.

        ProcessGraph may retain a ``LoopResult`` identity for a lexical read
        authored late in the loop body. Its SSA definition belongs at the
        exit, but that does not make the body read a future exit Phi: inside
        the loop the same spelling denotes the current ``updated_value_id``.
        Record and temporarily install that alias while the body and latch
        condition are lowered. The exit publication restores ownership of the
        result id afterward.
        """

        saved = []
        saved_ids: set[int] = set()
        for port_id, _initial_id, updated_id in getattr(loop, "result_ports", ()):
            port_id = int(port_id)
            updated_id = int(updated_id)
            # Include every pre-existing spelling between the public result
            # and its storage root. A composed graph may feed a body region
            # with an earlier spelling from this chain instead of the final
            # LoopResult id. The seen set makes malformed cyclic provenance a
            # finite walk; first discovery is the incumbent on a repeated id.
            spellings = []
            current = port_id
            seen: set[int] = set()
            while current not in seen:
                seen.add(current)
                spellings.append(current)
                if current not in self.value_aliases:
                    break
                current = int(self.value_aliases[current])
            for spelling in spellings:
                if spelling == updated_id or spelling in saved_ids:
                    continue
                saved_ids.add(spelling)
                saved.append((
                    spelling,
                    spelling in self.value_aliases,
                    self.value_aliases.get(spelling),
                ))
                # LOOP_BODY_SPELLING: DERIVED(the port's binding row, the
                # loop construct cell).
                self._post_alias(
                    spelling, updated_id, AliasKind.LOOP_BODY_SPELLING,
                    self._declared_cell(
                        LOOP_RESULT_PORT_BINDING.name,
                        (self.lexical_read_scope, int(port_id)),
                    ) if self.lexical_read_scope is not None else None,
                    self._construct_cell(getattr(loop, "source_loop_node_id", None)),
                    stage=CONTROL_SSA_LOOP,
                )
        return tuple(saved)

    def _restore_loop_result_port_aliases(
        self, saved: tuple[tuple[int, bool, int | None], ...],
        loop: Any = None,
    ) -> None:
        loop_cell = self._construct_cell(getattr(loop, "source_loop_node_id", None))
        for port_id, existed, previous in saved:
            # RESTORED: the undo is an edge from the saved PLANNING cell and
            # the loop cell, never an erasure.
            self._post_alias(
                int(port_id), None if not existed else int(previous),
                AliasKind.RESTORED, loop_cell, stage=CONTROL_SSA_LOOP,
            )

    def _post_alias(
        self, alias: Any, source_id: Any, kind: AliasKind, *cells: Any,
        stage: Any = None,
    ) -> Ref | None:
        """``control_value_alias`` row ``(scope, alias)``: a REVISE with
        ``AliasFact(source, kind)`` DERIVED from ``cells`` (and the previous
        revision), then the ``value_aliases`` read view."""

        alias = int(alias)
        if source_id is None:
            self.value_aliases.pop(alias, None)
        else:
            self.value_aliases[alias] = int(source_id)
        book = self._book()
        row = (self._scope(), alias)
        previous = book.latest_ref(CONTROL_VALUE_ALIAS, row)
        fact = AliasFact(None if source_id is None else int(source_id), kind)
        if previous is not None and (
            book.pages[CONTROL_VALUE_ALIAS.name].cells.get((row, previous.column))
            == fact
        ):
            return previous
        sources = self._cells(
            previous, *cells, self._canonical_cell(alias),
            None if source_id is None else self._canonical_cell(source_id),
        )
        if not sources:
            sources = (self._function_root(),)
        return book.post(
            CONTROL_VALUE_ALIAS, row, fact,
            stage=self._stage(stage),
            provenance=Derived(sources),
            mode=Mode.REVISE,
        )

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
            seed_value = self.fresh_value(
                dtype="float64", transform=CONTROL_CONST,
                operands=(seed_id, loop.source_loop_node_id),
            )
            self.emit(
                Handler.Const, [], seed_value,
                attributes={"value": float(seed_literal)},
            )
            # LOOP_SEED: the folded seed's own identity cell is the cause.
            self._bind(
                seed_id, seed_value, BindingKind.LOOP_SEED,
                self._construct_cell(loop.source_loop_node_id),
            )
        carried: list[tuple[int, int, SSAValue, SSAValue, SSAValue]] = []
        for updated_id, initial_id in loop.carried_aliases:
            updated_id = int(updated_id)
            initial_id = int(initial_id)
            # The concordance can prove that the initial, updated, and exit
            # values occupy one storage slot.  Their temporal SSA identities
            # remain distinct: the preheader must read the value that exists
            # before the loop, not its result port after the loop.
            initial_value = self.external_value(
                initial_id,
                # An initial with no binding of its own that the alias record
                # resolves (an in-place arena after an earlier loop's stores)
                # is that storage; not following it minted a second,
                # unconnected formal for the same array.
                follow_aliases=(
                    int(initial_id) not in self.external_values
                    and int(initial_id) in self.value_aliases
                ),
            )
            updated_value = self.external_values.get(updated_id)
            if updated_value is None:
                updated_value = SSAValue(
                    updated_id,
                    dtype=initial_value.dtype,
                    shape=initial_value.shape,
                )
                # LOOP_SEED: the reserved backedge slot, from the pre-loop
                # binding of the initial and the loop construct.
                self._bind(
                    updated_id, updated_value, BindingKind.LOOP_SEED,
                    initial_value, self._construct_cell(loop.source_loop_node_id),
                )
            else:
                # An authored binding may be updated to a value which
                # already exists before the loop (``last = proposal``).
                # That value is the exact backedge resident.  Reserving a
                # second SSA object under the same ProcessGraph id erases
                # the formal and later misclassifies the binding as an
                # identity update merely because no body instruction needs
                # to reproduce the already-dominating value.
                updated_value.accounting[
                    "loop_carried_preexisting_update"
                ] = True
            current_value = self.fresh_value(
                dtype=initial_value.dtype,
                shape=initial_value.shape,
                transform=PHI_LOOP_HEADER,
                operands=(initial_value, updated_value),
            )
            current_value.accounting.update({
                "source_value_id": initial_id,
                "carried_from_value_id": updated_id,
            })
            # Region output extraction will use this exact object as the
            # backedge definition referenced by the Phi below.
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

        loop_cell = self._construct_cell(loop.source_loop_node_id)
        induction = self.fresh_value(
            dtype="int", transform=PHI_LOOP_HEADER,
            operands=(start, stop, step, loop_cell),
        )
        next_induction = self.fresh_value(
            dtype="int", transform=CONTROL_EXPRESSION,
            operands=(induction, step),
        )
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
        bound_initial_ids: set[int] = set()
        for entry, (
            updated_id,
            initial_id,
            initial_value,
            updated_value,
            current_value,
        ) in enumerate(carried):
            self.emit(
                Handler.Phi,
                [initial_value, updated_value],
                current_value,
                attributes={
                    "incoming_blocks": (preheader.name, latch.name),
                    "binding": "loop_carried",
                    "initial_value_id": self._phi_initial_binding(
                        initial_id, initial_value, current_value,
                        stage=CONTROL_SSA_LOOP,
                    ),
                    "initial_spelled_value_id": int(initial_id),
                    "updated_value_id": updated_id,
                    "recursion_region_id": recursion_region_id,
                    # The loop this generation belongs to, so a consumer
                    # can reach the scope declaration without recovering
                    # it from branch topology a transformation may have
                    # rewritten.
                    "source_loop_node_id": loop.source_loop_node_id,
                },
            )
            carried_phis[entry] = self.current.instrs[-1]
            if initial_id not in bound_initial_ids:
                # LOOP_HEADER: the header Phi is the body's value of the
                # initial; its cause is the Phi cell and the loop construct.
                self._bind(
                    initial_id, current_value, BindingKind.LOOP_HEADER,
                    loop_cell,
                )
                bound_initial_ids.add(initial_id)
        # Declare the loop as an evolving scope.  The rebind table is
        # exactly what ``carried`` already holds; what was missing is any
        # statement that these three names are one quantity at three
        # generations, which is why three of them could disagree in
        # silence.
        try:
            from .identity_concordance import declare_loop_scope

            declare_loop_scope(
                self.function_name, loop.source_loop_node_id,
                header.name, latch.name, exit_block.name,
                self._loop_scope_rebinds(loop, carried),
            )
        except Exception:
            pass
        break_bound_initials = {
            int(initial_id): self._break_bound_initial(
                int(port_id), int(initial_id),
            )
            for port_id, initial_id, updated_id
            in getattr(loop, "result_ports", ())
            if int(updated_id) == int(initial_id)
        }
        result_port_aliases = self._bind_loop_result_ports_inside_body(loop)
        condition = self.fresh_value(
            dtype="bool", transform=CONTROL_PREDICATE,
            operands=(induction, stop),
        )
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
                (updated_id, initial_id, current, initial)
                for updated_id, initial_id, initial, _updated, current
                in carried
            ),
            # A port whose "updated" identity IS its initial identity has no
            # backedge version: the name is bound only on break paths.
            "break_bound": tuple(
                (int(port_id), int(initial_id))
                for port_id, initial_id, updated_id
                in getattr(loop, "result_ports", ())
                if int(updated_id) == int(initial_id)
            ),
            "break_edges": [],
            "continue_edges": [],
            "sites_seen": set(),
            "expected_sites": tuple(
                int(site) for site in getattr(loop, "control_site_ids", ())
            ),
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
                row_stride = self.external_value(pool.row_stride_value_id)
                base_offset = self.fresh_value(
                    dtype="int", transform=CONTROL_EXPRESSION,
                    operands=(handle, row_stride),
                )
                offset = self.fresh_value(
                    dtype="int", transform=CONTROL_EXPRESSION,
                    operands=(base_offset, induction),
                )
                address = self.fresh_value(
                    dtype="ptr", transform=ADDRESS, operands=(offset,),
                )
                target = self.produced_value(
                    int(target_id), dtype="unknown", cause=(address,),
                )
                self.emit(
                    Handler.Mul,
                    [handle, row_stride],
                    base_offset,
                    attributes={"binding": "child_table_offset"},
                )
                self.emit(
                    Handler.Add, [base_offset, induction], offset,
                    attributes={"binding": "child_table_offset"},
                )
                if pool.live_flags_value_id is not None:
                    live_address = self.fresh_value(
                        dtype="ptr", transform=ADDRESS, operands=(offset,),
                    )
                    live = self.fresh_value(
                        dtype="bool", transform=LOAD, operands=(live_address,),
                    )
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
                # LOOP_CONTROL: the induction Phi is the target's value.
                self._bind(
                    int(target_id), induction, BindingKind.LOOP_CONTROL,
                    loop_cell, int(iterable_id),
                )
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
                    target_shape = (
                        () if target_meta is None
                        else tuple(map(int, target_meta.shape or ()))
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
                        if target_shape:
                            source.shape = target_shape
                        source.accounting.update({
                            "projected_row_source_id": int(iterable_id),
                            "projected_row_column": int(column_projection),
                            **(
                                {"sequence_row_shape": target_shape}
                                if target_shape else {}
                            ),
                        })
                    else:
                        source = self.fresh_value(
                            dtype=target_dtype, shape=target_shape,
                            transform=ROW_COLUMN_PROJECTION,
                            operands=(int(iterable_id), int(target_id)),
                        )
                        source.accounting.update({
                            "projected_row_source_id": int(iterable_id),
                            "projected_row_column": int(column_projection),
                            **(
                                {"sequence_row_shape": target_shape}
                                if target_shape else {}
                            ),
                        })
                        self.arguments.append(source)
                    self.projected_row_columns[column_key] = source
                if int(target_id) in self.variant_projected_target_ids:
                    scalar_source = source
                    handle_source = self.variant_handle_columns.get(column_key)
                    if handle_source is None:
                        handle_source = self.fresh_value(
                            dtype="int", transform=ROW_COLUMN_PROJECTION,
                            operands=(int(iterable_id), scalar_source),
                        )
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
                    self._bind(
                        int(target_id), scalar_value, BindingKind.LOOP_CONTROL,
                        loop_cell,
                    )
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
            column_value = self.fresh_value(
                dtype="int", transform=CONTROL_CONST, operands=(base,),
            )
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
            aggregate = self.fresh_value(
                dtype="ssa.aggregate", transform=CONTROL_CONST,
                operands=(int(iterable_id), loop_cell),
            )
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
            closure_sources = [
                self.external_value(value_id) for value_id in source_ids
            ]
            aggregate = self.fresh_value(
                dtype="ssa.aggregate", transform=CONTROL_EXPRESSION,
                operands=(*closure_sources, int(aggregate_id)),
            )
            self.emit(
                Handler.Const,
                closure_sources,
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
        initial_counts: dict[int, int] = {}
        for _updated_id, initial_id, *_rest in carried:
            initial_counts[int(initial_id)] = (
                initial_counts.get(int(initial_id), 0) + 1
            )
        protected_alias_sources = frozenset(
            int(updated_id)
            for updated_id, initial_id, *_rest in carried
            if initial_counts[int(initial_id)] > 1
        )
        self.protected_loop_alias_sources.append(protected_alias_sources)
        self.enclosing_loop_states.append(
            self._enter_loop_state(loop, carried)
        )
        try:
            self.lower(loop.body, path=f"{path}.body")
            for mutation in loop.sequence_mutations:
                self.lower_sequence_mutation(mutation, path=path)
            for terminal in loop.terminal_controls:
                self.lower(terminal, path=f"{path}.terminal")
        finally:
            self.enclosing_loop_states.pop()
            self.protected_loop_alias_sources.pop()
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
        for entry, (updated_id, _initial_id, _initial, reserved, _current) in enumerate(carried):
            published = self.external_values.get(updated_id, reserved)
            carried_updates[entry] = published
            if published is not reserved:
                carried_phis[entry].args[1] = published
        if os.environ.get("TURING_DEBUG_LOOP_ALIAS_PUBLICATION"):
            print(
                "DEBUG-LOOP-CARRIED-UPDATES "
                f"fn={self.function_name} path={path} "
                f"updates={tuple((int(carried[entry][0]), int(value.id)) for entry, value in carried_updates.items())!r}",
                file=sys.stderr,
                flush=True,
            )
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
        for entry, (updated_id, _initial_id, _initial, updated, _current) in enumerate(carried):
            if id(carried_updates[entry]) not in produced_results:
                if updated.accounting.get(
                    "loop_carried_preexisting_update"
                ):
                    # Selecting a value which already dominates the loop is
                    # an authored update even though the body needs no new
                    # instruction to compute it.
                    continue
                declared_outputs = tuple(
                    region_index
                    for region_index, (_feeds, outputs)
                    in self.region_signatures.items()
                    if updated_id in outputs
                )
                alias_source = self.value_aliases.get(updated_id)
                concorded_resident = self._concorded_resident

                resident_id = concorded_resident(updated_id)
                identity_occurrences = tuple(dict.fromkeys((
                    int(updated_id), int(resident_id),
                    *(
                        int(value_id)
                        for value_id in self.region_value_meta
                        if concorded_resident(value_id) == resident_id
                    ),
                )))
                updated_meta = next((
                    self.region_value_meta.get(value_id)
                    for value_id in identity_occurrences
                    if tuple(getattr(
                        self.region_value_meta.get(value_id), "shape", ()
                    ) or ())
                ), None)
                storage_shape = tuple(
                    getattr(updated_meta, "shape", ()) or ()
                )
                from .identity_concordance import current_identity_book

                shape_page = current_identity_book().page(
                    "tensor_shape_concordance"
                )
                shape_contract = (
                    shape_page.latest((
                        self.tensor_shape_concordance_scope,
                        int(updated_id),
                    ))
                    or shape_page.latest((
                        self.tensor_shape_concordance_scope,
                        int(resident_id),
                    ))
                )
                if shape_contract is None and storage_shape:
                    shape_contract = {
                        "program_abi_storage": "span",
                        "program_abi_rank": len(storage_shape),
                        "tensor_metadata_state": "static",
                        "shape": storage_shape,
                        "source": "concorded-loop-resident-meta",
                    }
                    for value_id in dict.fromkeys((
                        int(resident_id), int(updated_id),
                    )):
                        shape_page.set(
                            (
                                self.tensor_shape_concordance_scope,
                                value_id,
                            ),
                            max(shape_page.columns, default=-1) + 1,
                            shape_contract,
                        )
                storage_rank = int(
                    (shape_contract or {}).get("program_abi_rank", 0) or 0
                )
                storage_is_span = (
                    (shape_contract or {}).get("program_abi_storage")
                    == "span"
                )
                if (
                    not declared_outputs
                    and alias_source is not None
                    and (
                        storage_shape
                        or storage_is_span
                        or storage_rank > 0
                        or int(alias_source) in self.sequence_descriptors
                    )
                ):
                    # Indexed stores mutate their resident span.  A nested
                    # retained loop can therefore publish no new pointer
                    # value even though its body produced every authored
                    # store.  Carry the current resident pointer across the
                    # enclosing latch, exactly as the while-loop path below
                    # does for a producerless identity backedge.  Restrict
                    # this to graph-proven shaped storage or a resident
                    # sequence descriptor built from the concorded contract:
                    # an ordinary scalar recurrence must still name a real
                    # producer.
                    current = carried_phis[entry].args[0]
                    carried_phis[entry].args[1] = current
                    carried_updates[entry] = current
                    # LOOP_LATCH: the resident pointer crosses the latch.
                    self._bind(
                        updated_id, current, BindingKind.LOOP_LATCH, loop_cell,
                    )
                    current.accounting["ssa_storage_identity_backedge"] = True
                    continue
                self.shortfalls.append(
                    SSALoweringShortfall(
                        "control",
                        "loop_carried",
                        f"{path}.body",
                        f"carried update value {updated_id} has no producer "
                        "inside the loop body; "
                        f"alias_source={alias_source!r}; "
                        f"declared_region_outputs={declared_outputs}",
                    )
                )
        for source_id, collection_id, induction_name, start in (
            self.program.collection_bindings
        ):
            if induction_name != loop.induction:
                continue
            publication_index = induction
            if str(start).strip() != "0":
                offset = self.expression_value(
                    str(start), location=f"{path}.collection-offset",
                )
                publication_index = self.fresh_value(
                    dtype="int", transform=CONTROL_EXPRESSION,
                    operands=(induction, offset),
                )
                self.emit(
                    Handler.Add,
                    [induction, offset],
                    publication_index,
                    attributes={"binding": "collection_offset"},
                )
            collection = self.external_value(collection_id)
            address = self.fresh_value(
                dtype="ptr", transform=ADDRESS,
                operands=(collection, publication_index),
            )
            self.emit(
                Handler.GetElementPtr,
                [
                    collection,
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
        self._complete_loop_latch_carried(
            carried,
            carried_phis,
            carried_updates,
            exit_context["continue_edges"],
        )
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
        # The targets leave scope at the loop's exit, and the exit is caused
        # by the latch: the induction update ``next_induction`` is what the
        # header compares to leave.  Its cell is the withdrawal's cause.
        # Without it the withdrawal cited exactly the cells the LOOP_CONTROL
        # binding cited (canonical id, induction, loop, iterable), and the
        # book refused it as a REVISE with no new source: measured on
        # ``for k in range(6)`` in the oscillator case's ``cos_series``,
        # graph id 0, ConcordanceRefusal at control_value_binding.
        exit_cause = self._value_cell(next_induction)
        for target_id, previous in restored_values.items():
            # RESTORED (or withdrawn): the loop targets leave scope.
            self._withdraw(target_id, previous, loop_cell, exit_cause)
        if previous_induction is None:
            self.local_control_values.pop(loop.induction, None)
        else:
            self.local_control_values[loop.induction] = previous_induction
        settled_initial_ids: set[int] = set()
        for updated_id, initial_id, _initial, _updated, current in carried:
            # LOOP_HEADER: after the loop every carried spelling denotes
            # the converged header Phi.
            if initial_id not in settled_initial_ids:
                self._bind(
                    initial_id, current, BindingKind.LOOP_HEADER, loop_cell,
                )
                settled_initial_ids.add(initial_id)
            self._bind(updated_id, current, BindingKind.LOOP_HEADER, loop_cell)
        for missing_site in sorted(
            set(exit_context["expected_sites"]) - exit_context["sites_seen"]
        ):
            # A source break/continue nothing placed (neither an arm nor the
            # lexical schedule) would silently turn into a loop that runs to
            # completion.
            self.shortfalls.append(SSALoweringShortfall(
                "control", "loop-control-site", path,
                f"source break/continue at graph node {missing_site} was "
                "never placed in the loop body",
            ))
        self._restore_loop_result_port_aliases(result_port_aliases, loop)
        self._publish_loop_result_ports(
            loop,
            header=header,
            exit_block=exit_block,
            carried=carried,
            break_edges=tuple(exit_context["break_edges"]),
            break_bound=tuple(exit_context["break_bound"]),
            break_bound_initials=break_bound_initials,
        )
        if deployment_id is not None:
            self.emit_deployment_boundary(Handler.Join, record)

    def lower_while(self, loop: WhileBlock, *, path: str) -> None:
        recursion_region_id = loop.recursion_region_id
        loop_cell = self._construct_cell(loop.source_loop_node_id)
        constant_predicate = (
            bool(loop.predicate_expression.literal)
            if (
                loop.predicate_expression is not None
                and loop.predicate_expression.op == "const"
                and isinstance(loop.predicate_expression.literal, bool)
            )
            else None
        )
        self.lower(loop.condition, path=f"{path}.condition.initial")
        preheader = self.current
        initial_predicate = (
            self.lower_control_expression(loop.predicate_expression)
            if loop.predicate_expression is not None
            else self.external_value(loop.predicate_value_id, dtype="bool")
        )
        next_predicate = self.fresh_value(
            dtype="bool", transform=CONTROL_PREDICATE,
            operands=(initial_predicate, loop_cell),
        )

        # A carried seed whose graph node was folded to a constant must enter
        # as that literal; external_value would otherwise invent a
        # producerless argument for it.
        for seed_id, seed_literal in getattr(loop, "carried_seeds", ()):
            seed_id = int(seed_id)
            if seed_id in self.external_values:
                continue
            seed_value = self.fresh_value(
                dtype="float64", transform=CONTROL_CONST,
                operands=(seed_id, loop.source_loop_node_id),
            )
            self.emit(
                Handler.Const, [], seed_value,
                attributes={"value": float(seed_literal)},
            )
            # LOOP_SEED: the folded seed's own identity cell is the cause.
            self._bind(
                seed_id, seed_value, BindingKind.LOOP_SEED,
                self._construct_cell(loop.source_loop_node_id),
            )
        carried: list[tuple[int, int, SSAValue, SSAValue, SSAValue]] = []
        for updated_id, initial_id in loop.carried_aliases:
            updated_id = int(updated_id)
            initial_id = int(initial_id)
            # Storage concordance does not erase the temporal distinction
            # between the preheader seed and the loop backedge value.
            initial_value = self.external_value(
                initial_id,
                # An initial with no binding of its own that the alias record
                # resolves (an in-place arena after an earlier loop's stores)
                # is that storage; not following it minted a second,
                # unconnected formal for the same array.
                follow_aliases=(
                    int(initial_id) not in self.external_values
                    and int(initial_id) in self.value_aliases
                ),
            )
            updated_value = self.external_values.get(updated_id)
            if updated_value is None:
                updated_value = SSAValue(
                    updated_id,
                    dtype=initial_value.dtype,
                    shape=initial_value.shape,
                )
                # LOOP_SEED: the reserved backedge slot, from the pre-loop
                # binding of the initial and the loop construct.
                self._bind(
                    updated_id, updated_value, BindingKind.LOOP_SEED,
                    initial_value, self._construct_cell(loop.source_loop_node_id),
                )
            else:
                # A carried binding can select an already-dominating value
                # as its authored update.  Keep that exact SSA resident;
                # manufacturing another object with the same graph id loses
                # the distinction between this binding and another binding
                # which merely shares its initial value.
                updated_value.accounting[
                    "loop_carried_preexisting_update"
                ] = True
            current_value = self.fresh_value(
                dtype=initial_value.dtype,
                shape=initial_value.shape,
                transform=PHI_LOOP_HEADER,
                operands=(initial_value, updated_value),
            )
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
        current_predicate = self.fresh_value(
            dtype="bool", transform=PHI_LOOP_HEADER,
            operands=(initial_predicate, next_predicate),
        )
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
        # LOOP_CONTROL: the predicate id denotes the header Phi in the body.
        self._bind(
            int(loop.predicate_value_id), current_predicate,
            BindingKind.LOOP_CONTROL, loop_cell,
        )
        carried_phis: dict[int, Instr] = {}
        bound_initial_ids: set[int] = set()
        for entry, (updated_id, initial_id, initial, updated, current) in enumerate(carried):
            self.emit(
                Handler.Phi,
                [initial, updated],
                current,
                attributes={
                    "incoming_blocks": (preheader.name, latch.name),
                    "binding": "loop_carried",
                    "initial_value_id": self._phi_initial_binding(
                        initial_id, initial, current,
                        stage=CONTROL_SSA_LOOP,
                    ),
                    "initial_spelled_value_id": int(initial_id),
                    "updated_value_id": updated_id,
                    "recursion_region_id": recursion_region_id,
                    # The loop this generation belongs to, so a consumer
                    # can reach the scope declaration without recovering
                    # it from branch topology a transformation may have
                    # rewritten.
                    "source_loop_node_id": loop.source_loop_node_id,
                },
            )
            carried_phis[entry] = self.current.instrs[-1]
            if initial_id not in bound_initial_ids:
                # LOOP_HEADER: the header Phi is the body's value.
                self._bind(
                    initial_id, current, BindingKind.LOOP_HEADER, loop_cell,
                )
                bound_initial_ids.add(initial_id)
        # Declare the loop as an evolving scope.  The rebind table is
        # exactly what ``carried`` already holds; what was missing is any
        # statement that these three names are one quantity at three
        # generations, which is why three of them could disagree in
        # silence.
        try:
            from .identity_concordance import declare_loop_scope

            declare_loop_scope(
                self.function_name, loop.source_loop_node_id,
                header.name, latch.name, exit_block.name,
                self._loop_scope_rebinds(loop, carried),
            )
        except Exception:
            pass
        break_bound_initials = {
            int(initial_id): self._break_bound_initial(
                int(port_id), int(initial_id),
            )
            for port_id, initial_id, updated_id
            in getattr(loop, "result_ports", ())
            if int(updated_id) == int(initial_id)
        }
        result_port_aliases = self._bind_loop_result_ports_inside_body(loop)
        if constant_predicate is None:
            self.conditional_branch(current_predicate, body, exit_block)
        else:
            self.branch(body if constant_predicate else exit_block)
            self.current.instrs[-1].attributes.update({
                "source_control": "while_constant_predicate",
                "source_value_id": int(loop.predicate_value_id),
                "constant_predicate": constant_predicate,
            })

        self.current = body
        self.loop_targets.append((latch, exit_block))
        exit_context = {
            "carried": tuple(
                (updated_id, initial_id, current, initial)
                for updated_id, initial_id, initial, _updated, current
                in carried
            ),
            # A port whose "updated" identity IS its initial identity has no
            # backedge version: the name is bound only on break paths.
            "break_bound": tuple(
                (int(port_id), int(initial_id))
                for port_id, initial_id, updated_id
                in getattr(loop, "result_ports", ())
                if int(updated_id) == int(initial_id)
            ),
            "break_edges": [],
            "continue_edges": [],
            "sites_seen": set(),
            "expected_sites": tuple(
                int(site) for site in getattr(loop, "control_site_ids", ())
            ),
        }
        self.loop_exit_contexts.append(exit_context)
        preserved_before_body = set(self.preserved_region_output_ids)
        self.preserved_region_output_ids.update(
            int(initial_id) for _updated_id, initial_id, *_rest in carried
        )
        blocks_before_body = {id(existing) for existing in self.blocks.values()}
        initial_counts: dict[int, int] = {}
        for _updated_id, initial_id, *_rest in carried:
            initial_counts[int(initial_id)] = (
                initial_counts.get(int(initial_id), 0) + 1
            )
        protected_alias_sources = frozenset(
            int(updated_id)
            for updated_id, initial_id, *_rest in carried
            if initial_counts[int(initial_id)] > 1
        )
        self.protected_loop_alias_sources.append(protected_alias_sources)
        loop_state = self._enter_loop_state(loop, carried)
        self.enclosing_loop_states.append(loop_state)
        try:
            self.lower(loop.body, path=f"{path}.body")
            for mutation in loop.sequence_mutations:
                self.lower_sequence_mutation(mutation, path=path)
            for terminal in loop.terminal_controls:
                self.lower(terminal, path=f"{path}.terminal")
        finally:
            self.enclosing_loop_states.pop()
            self.protected_loop_alias_sources.pop()
            self.preserved_region_output_ids = preserved_before_body
            self.loop_exit_contexts.pop()
            self.loop_targets.pop()
        body_blocks = [
            candidate
            for candidate in self.blocks.values()
            if id(candidate) not in blocks_before_body or candidate is body
        ]
        carried_updates: dict[int, SSAValue] = {}
        for entry, (updated_id, _initial_id, _initial, reserved, _current) in enumerate(carried):
            published = self.external_values.get(updated_id, reserved)
            carried_updates[entry] = published
            if published is not reserved:
                carried_phis[entry].args[1] = published
        if os.environ.get("TURING_DEBUG_LOOP_ALIAS_PUBLICATION"):
            print(
                "DEBUG-WHILE-CARRIED-UPDATES "
                f"fn={self.function_name} path={path} "
                f"updates={tuple((int(carried[entry][0]), int(value.id)) for entry, value in carried_updates.items())!r}",
                file=sys.stderr,
                flush=True,
            )
        produced_results = {
            id(instruction.res)
            for basic_block in body_blocks
            for instruction in basic_block.instrs
            if instruction.res is not None
        }
        for entry, (updated_id, _initial_id, _initial, updated, current) in enumerate(carried):
            if id(carried_updates[entry]) not in produced_results:
                if updated.accounting.get(
                    "loop_carried_preexisting_update"
                ):
                    # The update is an input or another value defined before
                    # the loop, so it correctly dominates the latch without
                    # a producer in the body.  It is not an untouched carried
                    # name and must not be replaced by the header Phi.
                    continue
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
                    carried_phis[entry].args[1] = current
                    carried_updates[entry] = current
                    # LOOP_LATCH: the header Phi is the identity backedge.
                    self._bind(
                        updated_id, current, BindingKind.LOOP_LATCH, loop_cell,
                    )
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
        self._complete_loop_latch_carried(
            carried,
            carried_phis,
            carried_updates,
            exit_context["continue_edges"],
        )
        # The latch evaluates the guard for the NEXT iteration, so every
        # carried name must resolve to the body's UPDATED value there --
        # not the header phi it is bound to for the rest of the loop.
        # Testing the pre-update value lags the guard one iteration behind
        # and runs the loop once too often (scorecard level 17: the
        # compiled while halved 1.0 to 0.5 where the authored loop stops).
        latch_restore = []
        latch_initial_ids: set[int] = set()
        for updated_id, initial_id, _initial, updated, _current in carried:
            if initial_id in latch_initial_ids:
                continue
            latch_initial_ids.add(initial_id)
            latch_restore.append(
                (initial_id, self.external_values.get(initial_id))
            )
            # LOOP_LATCH: at the latch the carried name is the body's update.
            self._bind(
                initial_id, self.external_values.get(updated_id, updated),
                BindingKind.LOOP_LATCH, self._binding_cell(updated_id), loop_cell,
            )
        # A retained numeric condition and its explicit scalar expression can
        # both publish the predicate. They must not define the same SSA id.
        # The expression owns the backedge value; numeric projection gets a
        # separate result when both representations are present.
        carried_test = (
            loop.predicate_expression is not None
            and loop.predicate_expression.op == "value"
            and loop.predicate_expression.value_id is not None
            and any(
                int(initial_id) == int(loop.predicate_expression.value_id)
                for _updated_id, initial_id, *_rest in carried
            )
        )
        if carried_test and loop_state is not None:
            # Decision recorded, not held: row ``(control scope, loop)`` ->
            # the carried binding the test reads (``loop_carried_entry``
            # names its entry).  The latch neither re-runs the pre-loop
            # condition regions nor re-publishes the predicate id for it.
            # DERIVED(the ``lexical_read_binding`` cells the predicate read);
            # a bare predicate id with no read position is recorded as
            # ``Unresolved(WHILE_TEST_NO_READ_EXPRESSION)``.
            predicate_cell = self._canonical_cell(
                loop.predicate_expression.value_id
            )
            if loop.predicate_expression.read is not None:
                read_cells = self._read_binding_cells(
                    (tuple(loop.predicate_expression.read),)
                )
                test_fact: Any = tuple(sorted(map(str, self._operand_bindings(
                    (tuple(loop.predicate_expression.read),)
                ))))
                test_sources = self._cells(*read_cells, predicate_cell, loop_cell)
            else:
                test_fact = Unresolved(
                    WHILE_TEST_NO_READ_EXPRESSION,
                    read=self._cells(predicate_cell),
                )
                test_sources = self._cells(predicate_cell, loop_cell)
            if not test_sources:
                test_sources = (self._function_root(),)
            self._book().post(
                WHILE_CARRIED_TEST, loop_state, test_fact,
                stage=self.active_stage,
                provenance=Derived(test_sources),
                mode=Mode.CONCORD,
            )
        if not carried_test:
            # A carried test's id is its binding's initial, which the latch
            # has just rebound to the body's update; that is the next test.
            self._bind(
                int(loop.predicate_value_id),
                self.fresh_value(
                    dtype="bool", transform=CONTROL_PREDICATE,
                    operands=(current_predicate, loop_cell),
                )
                if loop.predicate_expression is not None else next_predicate,
                BindingKind.LOOP_CONTROL, loop_cell,
            )
        preserved_before = set(self.preserved_region_output_ids)
        self.preserved_region_output_ids.update(
            int(initial_id) for _updated_id, initial_id, *_rest in carried
        )
        # The latch re-evaluates the predicate for the next iteration with
        # carried bindings at their updated values; every other binding that
        # shares a carried initial id still reads its pre-loop value, so the
        # loop's frame governs these reads exactly as it governs the body.
        self.enclosing_loop_states.append(loop_state)
        latch_leaf = None
        latch_frame = self.loop_frames.get(loop_state)
        if latch_frame is not None:
            latch_frame["latch"] = True
        try:
            if not carried_test:
                # A carried test (``while go:``) has no latch computation:
                # the condition regions produced its pre-loop value, and
                # re-running them would test that value forever.
                self.lower(loop.condition, path=f"{path}.condition.latch")
            if loop.predicate_expression is not None:
                latch_predicate = self.lower_control_expression(
                    loop.predicate_expression,
                    result_override=next_predicate,
                )
                if latch_predicate is not next_predicate:
                    # A bare leaf (``while go:``) computes nothing: the next
                    # test IS the value the leaf resolves to at the latch.
                    header_condition = next(
                        instruction for instruction in header.instrs
                        if instruction.res is current_predicate
                    )
                    header_condition.args[1] = latch_predicate
                    latch_leaf = latch_predicate
        finally:
            if latch_frame is not None:
                latch_frame["latch"] = False
            self.enclosing_loop_states.pop()
            self.preserved_region_output_ids = preserved_before
        # Post-loop consumers read the converged header phi, not the last
        # body update: restore the loop-wide binding before leaving.
        for initial_id, previous in latch_restore:
            # RESTORED: post-loop consumers read the header Phi again.
            self._withdraw(initial_id, previous, loop_cell)
        if latch_leaf is None and not any(
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
        settled_initial_ids: set[int] = set()
        for updated_id, initial_id, _initial, _updated, current in carried:
            # LOOP_HEADER: after the loop every carried spelling denotes
            # the converged header Phi.
            if initial_id not in settled_initial_ids:
                self._bind(
                    initial_id, current, BindingKind.LOOP_HEADER, loop_cell,
                )
                settled_initial_ids.add(initial_id)
            self._bind(updated_id, current, BindingKind.LOOP_HEADER, loop_cell)
        self._bind(
            int(loop.predicate_value_id), current_predicate,
            BindingKind.LOOP_CONTROL, loop_cell,
        )
        for missing_site in sorted(
            set(exit_context["expected_sites"]) - exit_context["sites_seen"]
        ):
            # A source break/continue nothing placed (neither an arm nor the
            # lexical schedule) would silently turn into a loop that runs to
            # completion.
            self.shortfalls.append(SSALoweringShortfall(
                "control", "loop-control-site", path,
                f"source break/continue at graph node {missing_site} was "
                "never placed in the loop body",
            ))
        self._restore_loop_result_port_aliases(result_port_aliases, loop)
        self._publish_loop_result_ports(
            loop,
            header=header,
            exit_block=exit_block,
            carried=carried,
            break_edges=tuple(exit_context["break_edges"]),
            break_bound=tuple(exit_context["break_bound"]),
            break_bound_initials=break_bound_initials,
        )

    def lower_state_machine(
        self,
        tick: StateMachineTick,
        *,
        path: str,
    ) -> None:
        if tick.state_value_id is not None:
            # The state is the graph value the selector reads, found through
            # the book like a conditional's predicate -- never a name looked
            # up in the uniform table.
            state = self.external_value(int(tick.state_value_id))
        else:
            state = self.uniform_values.get(str(tick.state))
            if state is None:
                state = self.expression_value(
                    tick.state,
                    location=f"{path}.state",
                )
        merge = self.new_block("state_merge")
        literal_ids = tuple(tick.case_value_ids) + (None,) * (
            len(tick.cases) - len(tick.case_value_ids)
        )
        aliases = tuple(tick.carried_aliases)
        # Each arm starts from the versions the tick was entered with: take
        # the snapshot of every carried initial before any arm lowers, and
        # restore it between arms (a nested merge inside an arm would
        # otherwise leave its result as the next arm's incumbent).  One
        # ``carried_snapshot`` row per id, DERIVED from the binding cell
        # current at entry and the tick's construct cell.
        carried_snapshots = {
            int(initial_id): self.external_value(int(initial_id))
            for _arms, _default, initial_id, _merged in aliases
        }
        tick_cell = self._construct_cell(tick.source_node_id)
        snapshot_cells = (
            self._post_carried_snapshots(tick_cell, carried_snapshots)
            if aliases else {}
        )
        stage_before = self.active_stage
        if aliases:
            self.active_stage = CONTROL_SSA_CONDITIONAL

        def restore_snapshots() -> None:
            for snapshot_id, snapshot_value in carried_snapshots.items():
                if self.external_values.get(int(snapshot_id)) is not snapshot_value:
                    self._bind(
                        int(snapshot_id), snapshot_value, BindingKind.RESTORED,
                        snapshot_cells.get(int(snapshot_id)), tick_cell,
                    )

        # (exit block, {initial id: (arm value, source cell)}) per arm that
        # reaches the merge.
        reaching: list[tuple[BasicBlock, dict[int, tuple[SSAValue, Any]]]] = []

        def arm_values(arm_slot: int | None, where: str) -> dict:
            found = {}
            for arm_ids, default_id, initial_id, _merged in aliases:
                arm_id = (
                    int(default_id) if arm_slot is None
                    else int(arm_ids[arm_slot])
                )
                found[int(initial_id)] = self._carried_name_arm(
                    arm_id, initial_id, carried_snapshots[int(initial_id)],
                    snapshot_cells.get(int(initial_id)), path=where,
                )
            return found

        for index, (case_value, case_body) in enumerate(tick.cases):
            # The case literal is a graph node with an identity cell: the
            # case's blocks and the literal value derive from it.  A case
            # with no node (a hand-built tick) is a bare spelling.
            literal_cell = (
                None if literal_ids[index] is None
                else self._canonical_cell(literal_ids[index])
            )
            case = self.new_block("state_case", also=(literal_cell,))
            otherwise = self.new_block("state_next", also=(literal_cell,))
            literal_int = _case_literal_int(case_value)
            literal = (
                self.constant_value(literal_int, literal_cell)
                if literal_cell is not None and literal_int is not None
                else self.expression_value(
                    case_value,
                    location=f"{path}.case[{index}]",
                )
            )
            condition = self.fresh_value(
                dtype="bool", transform=CONTROL_PREDICATE,
                operands=(state, literal),
            )
            self.emit(Handler.Eq, [state, literal], condition)
            self.conditional_branch(condition, case, otherwise)
            self.current = case
            restore_snapshots()
            values_before_arm = dict(self.external_values)
            self.lower(case_body, path=f"{path}.case[{index}].body")
            falls_through = (
                not self.current.successors
                or merge.name in self.current.successors
            )
            exit_block = self.current
            carried_here = (
                arm_values(index, f"{path}.case[{index}].body")
                if aliases else {}
            )
            if not self.current.successors:
                self.branch(merge)
            if falls_through:
                reaching.append((exit_block, carried_here))
            else:
                # An arm that leaves through its own edge (break / continue /
                # return) never reaches the merge: nothing it binds may.
                self._restore_view(values_before_arm, tick_cell)
            self.current = otherwise
        restore_snapshots()
        if tick.default is not None:
            values_before_arm = dict(self.external_values)
            self.lower(tick.default, path=f"{path}.default")
            falls_through = (
                not self.current.successors
                or merge.name in self.current.successors
            )
            exit_block = self.current
            carried_here = (
                arm_values(None, f"{path}.default") if aliases else {}
            )
            if not self.current.successors:
                self.branch(merge)
            if falls_through:
                reaching.append((exit_block, carried_here))
            else:
                self._restore_view(values_before_arm, tick_cell)
        else:
            # No default arm: a state matching no case falls through with
            # the versions the tick was entered with.
            exit_block = self.current
            carried_here = {
                initial_id: (
                    carried_snapshots[initial_id],
                    snapshot_cells.get(initial_id),
                )
                for initial_id in carried_snapshots
            }
            if not self.current.successors:
                self.branch(merge)
                reaching.append((exit_block, carried_here))
        self.current = merge
        if reaching:
            for arm_ids, default_id, initial_id, merged_id in aliases:
                self._publish_state_merge(
                    state, reaching, carried_snapshots, snapshot_cells,
                    tick_cell, arm_ids, default_id, initial_id, merged_id,
                )
        self.active_stage = stage_before

    def _publish_state_merge(
        self,
        state: SSAValue,
        reaching: list,
        carried_snapshots: Mapping[int, SSAValue],
        snapshot_cells: Mapping[int, Any],
        tick_cell: Any,
        arm_ids: tuple[int, ...],
        default_id: int,
        initial_id: int,
        merged_id: int,
    ) -> None:
        """One Phi joining every arm that reaches a tick's merge for one
        carried scalar, the N-way counterpart of ``lower_conditional``'s
        ``conditional_carried`` Phi.  Its value is made from each arm's
        source cell (its binding, or the snapshot when it did not write),
        the snapshot cell and the state it is selected by; every id taking
        part in the join denotes the merged version afterwards
        (``CONDITIONAL_MERGE``)."""

        initial_id = int(initial_id)
        initial = carried_snapshots[initial_id]
        incoming = [values[initial_id] for _block, values in reaching]
        incoming_values = [value for value, _source in incoming]
        sources = [source for _value, source in incoming]
        phi_operands = (
            *sources, snapshot_cells.get(initial_id), *incoming_values, state,
        )
        if int(merged_id) in {int(value.id) for value in incoming_values}:
            # The graph reuses an arm's write id as the join id: they are the
            # same source version but the join is a distinct SSA definition.
            merged = self.fresh_value(
                dtype=initial.dtype, shape=initial.shape,
                transform=PHI_CONDITIONAL, operands=phi_operands,
            )
            merged.accounting.update({
                "source_value_id": int(merged_id),
                "ssa_conditional_write_version": True,
            })
        else:
            merged = SSAValue(
                int(merged_id), dtype=initial.dtype, shape=initial.shape,
            )
            self._value_cell(merged, *phi_operands)
        self.emit(
            Handler.Phi, incoming_values, merged,
            attributes={
                "incoming_blocks": tuple(block.name for block, _ in reaching),
                "binding": "state_carried",
                "initial_value_id": initial_id,
            },
        )
        protected_sources = frozenset().union(
            *self.protected_loop_alias_sources
        ) if self.protected_loop_alias_sources else frozenset()
        published = {int(merged_id): merged}
        for value_id in (initial_id, int(default_id), *map(int, arm_ids)):
            if value_id not in protected_sources:
                published[value_id] = merged
        self._rebind_view(
            published, BindingKind.CONDITIONAL_MERGE, merged, tick_cell,
        )

    def _finish_pages(
        self,
        named_returns: tuple,
        parameter_value_names: tuple,
        returned: tuple,
    ) -> dict[str, Any]:
        """Post ``function_parameter`` / ``function_output`` and materialize
        the metadata tuples from the pages (plan 80 B2.6).

        ``parameter_names`` and ``named_outputs`` are read back from the rows
        just posted; ``carried_port_values`` and ``value_aliases`` from the
        ``carried_port_value`` and ``control_value_alias`` pages.  Each is
        checked against the value the builder computed and the computed
        value is kept when a page cannot reproduce it (a duplicate returned
        id, a port whose object the table lost), so the metadata the 18
        readers consume stays byte-identical to today.
        """

        book = self._book()
        scope = self._scope()
        # -- function_parameter: DERIVED(declared_parameter USED, the
        #    argument's ssa_value cell, its binding cell).
        for name, value_id in parameter_value_names:
            declared = self._declare_parameter(value_id, ParameterDeclaration.USED)
            argument = self.ssa_value_objects.get(int(value_id))
            self._post_concord(
                FUNCTION_PARAMETER, (scope, str(name)), int(value_id),
                declared, self._binding_cell(value_id), argument,
                stage=CONTROL_SSA_FINISH,
            )
        page = book.pages.get(FUNCTION_PARAMETER.name)
        parameter_names = tuple(
            (str(row[1]), int(page.latest(row)))
            for row in (() if page is None else page.scope_rows(scope))
            if page.latest(row) is not None
        )
        if parameter_names != tuple(parameter_value_names):
            parameter_names = tuple(parameter_value_names)
        # -- function_output: slot -> (name or None, ssa id), DERIVED(the
        #    slot value's cell (the merge Phi or the single edge's value)).
        pending_names = list(named_returns)
        for slot, value in enumerate(returned):
            name = next(
                (
                    candidate for candidate, value_id in pending_names
                    if int(value_id) == int(value.id)
                ),
                None,
            )
            if name is not None:
                pending_names.remove((name, next(
                    value_id for candidate, value_id in pending_names
                    if candidate == name and int(value_id) == int(value.id)
                )))
            self._post_concord(
                FUNCTION_OUTPUT, (scope, int(slot)), (name, int(value.id)),
                value, self._binding_cell(value.id), stage=CONTROL_SSA_FINISH,
            )
        page = book.pages.get(FUNCTION_OUTPUT.name)
        named_outputs = tuple(
            (str(fact[0]), int(fact[1]))
            for row in (() if page is None else page.scope_rows(scope))
            for fact in (page.latest(row),)
            if fact is not None and fact[0] is not None
        )
        if named_outputs != tuple(named_returns):
            named_outputs = tuple(named_returns)
        # -- carried_port_values: port -> the Phi object the page's cell names.
        legacy_ports = dict(getattr(self, "_carried_port_values", {}) or {})
        page = book.pages.get(CARRIED_PORT_VALUE.name)
        carried_port_values: dict[int, SSAValue] = {}
        for row in (() if page is None else page.scope_rows(scope)):
            fact = page.latest(row)
            if not isinstance(fact, Ref):
                continue
            port = self.ssa_value_objects.get(int(fact.row[1]))
            if port is not None:
                carried_port_values[int(row[1])] = port
        if carried_port_values != legacy_ports:
            carried_port_values = legacy_ports
        # -- value_aliases: alias -> source, the latest alias rows.
        page = book.pages.get(CONTROL_VALUE_ALIAS.name)
        value_aliases: dict[int, int] = {}
        for row in (() if page is None else page.scope_rows(scope)):
            fact = page.latest(row)
            if isinstance(fact, AliasFact) and fact.source_id is not None:
                value_aliases[int(row[1])] = int(fact.source_id)
        if value_aliases != dict(self.value_aliases):
            value_aliases = dict(self.value_aliases)
        return {
            "parameter_names": parameter_names,
            "named_outputs": named_outputs,
            "carried_port_values": carried_port_values,
            "value_aliases": value_aliases,
        }

    def finish(self) -> tuple[Function, tuple[SSALoweringShortfall, ...]]:
        self.active_stage = CONTROL_SSA_FINISH
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

        def owned_literal(value_id: int) -> SSAValue | None:
            """Resolve an output whose identity is a canonical literal.

            ``z = 0; t = x + z; return t, z``: no region publishes ``z``.
            The planner keeps every region's rematerialised copy private
            (``control_owned_literals``) and the control function owns the
            literal, so resolve it as the provisional formal that
            ``_materialize_control_constants`` turns into this function's
            own ``Const`` -- exactly how a folded ``flag = True; break`` is
            resolved on its break edge.  Silently dropping it here left the
            output to the call-frame pass, which found the callee's private
            copy and appended it to the region call's declared outputs after
            the callee's Ret was already fixed: on the vehicle body, 144
            declared against 143 produced, the literal's id last.
            """
            if int(value_id) not in self.constant_value_ids:
                return None
            return self.external_value(int(value_id))

        for name, history in self.named_output_histories.items():
            if self.function_return_edges:
                # With explicit return edges, source history is the temporal
                # authority for the still-open lexical fallthrough. Resolve
                # each version completely before considering an older one: a
                # newest owned literal (especially ``None``) must not lose to
                # an already-materialized value from an earlier return.
                value = next((
                    resolved
                    for value_id in reversed(history)
                    for resolved in (
                        existing_value(value_id) or owned_literal(value_id),
                    )
                    if resolved is not None
                ), None)
            else:
                # In a single-exit conditional, histories contain competing
                # arm assignments rather than a temporal fallthrough. Prefer
                # the value materialized by control lowering; only use an
                # owned literal if no arm/join value exists.
                value = next((
                    resolved
                    for value_id in reversed(history)
                    for resolved in (existing_value(value_id),)
                    if resolved is not None
                ), None)
                if value is None:
                    value = next((
                        resolved
                        for value_id in reversed(history)
                        for resolved in (owned_literal(value_id),)
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
            if value is None:
                value = owned_literal(value_id)
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
        # The function's ``Ret`` is emitted after this naming pass, so a
        # parameter whose only use is being returned (``return vehicle_input,
        # ...``) is not yet visible in any block.  Returning an argument is a
        # real ABI use: the caller must supply it, so it must keep its name.
        used_value_ids.update(int(value.id) for value in returned)
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
        fallthrough_open = not self.current.successors and (
            not self.current.instrs
            or self.current.instrs[-1].op
            not in {Handler.Br.value, Handler.CondBr.value, Handler.Ret.value}
        )
        if self.function_return_edges:
            # Control-aware result merging: every source ``return`` below the
            # top level branched to `function_exit` carrying its own slot
            # values; the lexical fall-through epilogue (if still open) is
            # one more edge carrying the values computed above. One Phi per
            # output slot, one Ret -- never "the last return's values on
            # every path".
            if os.environ.get("TURING_DEBUG_CONTROL_OVERLAY"):
                for block, values in self.function_return_edges:
                    print(
                        f"DEBUG-RETURN-EDGE fn={self.function_name} "
                        f"block={block.name} "
                        f"values={[None if v is None else (int(v.id), str(v.dtype), sorted((v.accounting or {}).keys())) for v in values]!r}",
                        file=sys.stderr,
                    )
            slot_count = max(
                len(named_returns),
                *(len(values) for _block, values in self.function_return_edges),
            )
            edges = list(self.function_return_edges)
            if fallthrough_open:
                fallthrough_values = tuple(
                    returned[index] if index < len(returned) else None
                    for index in range(slot_count)
                )
                edges.append((self.current, fallthrough_values))
                self.branch(self.function_exit_block())
            exit_block = self.function_exit_block()
            self.current = exit_block
            merged_values: list[SSAValue] = []
            slot_names = [name for name, _value_id in named_returns]
            for slot in range(slot_count):
                incoming_blocks = []
                incoming_values = []
                for block, values in edges:
                    value = values[slot] if slot < len(values) else None
                    if value is None:
                        # An unresolved slot on this return: keep the Phi
                        # arity honest with an explicit absence and record
                        # the shortfall rather than silently dropping the edge.
                        value = self.fresh_value(
                            dtype="none", transform=CONTROL_CONST,
                        )
                        self.current = block
                        # Insert before the block's terminator.
                        terminator = block.instrs.pop()
                        self.emit(Handler.NoneValue, [], value)
                        block.instrs.append(terminator)
                        self.current = exit_block
                        self.shortfalls.append(SSALoweringShortfall(
                            "control", "return",
                            f"{self.function_name}.return_slot[{slot}]",
                            "a return statement's slot value could not be "
                            f"resolved on edge {block.name!r}",
                        ))
                    incoming_blocks.append(block.name)
                    incoming_values.append(value)
                dtype = next(
                    (str(value.dtype) for value in incoming_values
                     if value.dtype not in {None, "unknown", "none"}),
                    str(incoming_values[0].dtype or "unknown"),
                )
                if len(edges) == 1:
                    merged = incoming_values[0]
                else:
                    merged = self.fresh_value(
                        dtype=dtype, transform=PHI_RETURN_MERGE,
                        operands=tuple(incoming_values),
                    )
                    self.emit(
                        Handler.Phi,
                        incoming_values,
                        merged,
                        attributes={
                            "incoming_blocks": tuple(incoming_blocks),
                            "binding": "return_merge",
                            "return_slot_index": slot,
                            **(
                                {"output_name": slot_names[slot]}
                                if slot < len(slot_names) else {}
                            ),
                        },
                    )
                merged_values.append(merged)
            returned = merged_values
            named_returns = [
                (name, int(merged_values[index].id))
                for index, (name, _value_id) in enumerate(named_returns)
                if index < len(merged_values)
            ]
            self.emit(Handler.Ret, returned)
        elif fallthrough_open:
            self.emit(Handler.Ret, returned)
        page_metadata = self._finish_pages(
            tuple(named_returns), tuple(parameter_value_names), tuple(returned),
        )
        function = Function(
                self.function_name,
                self.arguments,
                self.blocks,
                metadata={
                    "recursion_table": dict(self.ssa_recursion_table),
                    "named_outputs": page_metadata["named_outputs"],
                    # port id -> the carried phi VALUE standing at that
                    # port after the loops.  The record-return expansion
                    # resolves layout components by id, and a component
                    # whose id doubles as a written field slot must resolve
                    # to the phi, not the unwritten slot argument.
                    "carried_port_values": page_metadata["carried_port_values"],
                    "value_aliases": page_metadata["value_aliases"],
                    "value_names": tuple(value_names),
                    "parameter_names": page_metadata["parameter_names"],
                    "validation_contracts": tuple(self.validation_contracts),
                    "control_identity_receipts": tuple(dict.fromkeys(
                        self.control_identity_receipts
                    )),
                    "table_lookup_ownership_receipts": tuple(dict.fromkeys(
                        self.table_lookup_ownership_receipts
                    )),
                    "control_ir": True,
                    "deployment_regions": tuple(deployment_regions),
                    "sequence_table": SSASequenceTable(
                        dict(self.sequence_descriptors),
                        owner=self.function_name,
                    ),
                    "sequence_static_capacity_bounds": tuple(sorted(
                        self.static_sequence_capacities.items()
                    )),
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
            _canonicalize_non_dominating_loop_result_uses(
                function,
                book_scope=str(self.tensor_shape_concordance_scope),
            )
        )
        if loop_result_rebindings:
            function.metadata["loop_result_use_rebindings"] = (
                loop_result_rebindings
            )
        if self.evolution is not None and self.ssa_evolution is not None:
            self.evolution.bind_artifact(function, self.ssa_evolution)
            self.evolution.close_graph(self.ssa_evolution)
        return function, tuple(self.shortfalls)


def _post_loop_result_reconciliation(
    book: Any, row: tuple, fact: tuple, sources: tuple,
) -> None:
    """``loop_result_reconciliation`` row ``(function, argument)``: REVISE,
    DERIVED(the cells the pass read -- see ``_note``).

    A pass that runs again over the same, unchanged cells and states the same
    thing adds nothing.  One that states something different about unchanged
    cells (the IR was rewritten between the runs by a pass whose cell is not
    in hand here) has no cause on the book: the api refuses the derived
    revision, and the statement is posted ``Unsourced`` with the reason, so
    the audit lists it instead of the page dropping it.  A value with no
    cell at all is posted ``Unsourced`` the same way.
    """

    stored = book.pages.get(LOOP_RESULT_RECONCILIATION.name)
    if sources:
        try:
            book.post(
                LOOP_RESULT_RECONCILIATION, row, fact,
                stage=CONTROL_SSA_FINISH, provenance=Derived(sources),
                mode=Mode.REVISE,
            )
            return
        except ConcordanceRefusal:
            if stored is not None and stored.latest(row) == fact:
                return
            reason = REVISION_CAUSE_NOT_ON_BOOK
    else:
        reason = SSA_VALUE_NOT_ON_BOOK
    book.post(
        LOOP_RESULT_RECONCILIATION, row, fact, stage=CONTROL_SSA_FINISH,
        provenance=Unsourced(reason), mode=Mode.REVISE,
    )


def _canonicalize_non_dominating_loop_result_uses(
    function: Function,
    book_scope: str | None = None,
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
    # An empty ledger is not evidence of nothing to do.  A single-incoming
    # exit Phi states its own equivalence, and the dominance checks below
    # still decide every substitution, so the pass runs on CFG evidence
    # whether or not a recorded ledger accompanies it.
    if not function.blocks:
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
            import os as _rc_os
            if _rc_os.environ.get("TURING_TRACE_RECONCILE") and str(
                block_name
            ) == "loop_body" and "lu_decompose" in str(function.name):
                print("RC", str(function.name)[-14:], block_name,
                      instruction_index, instruction.op,
                      [int(getattr(a, "id", -1)) for a in instruction.args][:8],
                      flush=True)
            for argument_index, argument in enumerate(instruction.args):
                def _note(
                    outcome: str, detail: Any = None,
                    candidate: Any = None,
                ) -> None:
                    """Record this argument's reconciliation decision."""

                    try:
                        from .identity_concordance import (
                            current_identity_book,
                        )

                        book = current_identity_book()
                        row = (
                            str(function.name), int(argument.id),
                        )
                        # At the builder's finish the function does not yet
                        # carry its metadata scope; the builder passes the
                        # scope its own ``ssa_value`` cells were posted under.
                        scope = str(
                            book_scope
                            or function.metadata.get(
                                "tensor_shape_concordance_scope"
                            )
                            or function.name
                        )
                        # The rows it read: the argument's own ``ssa_value``
                        # cell, the consumer's result cell (each use is a
                        # different consumer, so each use is a different
                        # cause for the next revision), and the replacement's
                        # cell when it replaced.
                        sources = tuple(dict.fromkeys(
                            cell for cell in (
                                book.latest_ref(
                                    CARRIED_PORT_VALUE, (scope, int(argument.id)),
                                ),
                                book.latest_ref(
                                    CONTROL_VALUE_BINDING, (scope, int(argument.id)),
                                ),
                                book.latest_ref(
                                    SSA_VALUE, (scope, int(argument.id)),
                                ),
                                *(
                                    (book.latest_ref(
                                        SSA_VALUE, (scope, int(instruction.res.id)),
                                    ),)
                                    if instruction.res is not None else ()
                                ),
                                book.latest_ref(
                                    SSA_BLOCK,
                                    (scope, str(function.name), str(block_name)),
                                ),
                                *(
                                    (book.latest_ref(
                                        SSA_VALUE, (scope, int(candidate.id)),
                                    ),)
                                    if candidate is not None else ()
                                ),
                            )
                            if cell is not None
                        ))
                        fact = (
                            outcome,
                            f"{block_name}#{instruction_index}",
                            str(instruction.op),
                            detail,
                        )
                        _post_loop_result_reconciliation(
                            book, row, fact, sources,
                        )
                    except Exception:
                        pass

                replacement = port_values.get(int(argument.id))
                if replacement is argument:
                    # A ledger entry mapping a port to ITSELF is not an
                    # equivalence, and taking it as one made this pass decline
                    # silently: the single case it exists to repair -- a loop
                    # body reading its own result port -- is exactly the case
                    # the ledger records that way.  Treat it as no recorded
                    # equivalence and let the CFG supply the candidate.
                    replacement = None
                if replacement is None:
                    # A result port with no recorded equivalence still states
                    # one itself: an exit Phi with a SINGLE incoming says the
                    # port IS that incoming value.  A use the exit cannot
                    # dominate -- the loop body reading its own result -- may
                    # therefore read the incoming instead.  The dominance
                    # checks below still decide; this only supplies the
                    # candidate the recorded ledger was missing.
                    sites = definition_sites.get(int(argument.id), ())
                    if len(sites) == 1:
                        site_block, site_index = sites[0]
                        defining = function.blocks[site_block].instrs[
                            site_index
                        ]
                        if (
                            str(defining.op).casefold() == "phi"
                            and len(defining.args) == 1
                        ):
                            replacement = defining.args[0]
                        elif (
                            str(defining.op).casefold() == "phi"
                            and defining.args
                            and all(
                                int(other.id) == int(defining.args[0].id)
                                for other in defining.args
                            )
                            and int(defining.args[0].id) != int(argument.id)
                        ):
                            # A Phi whose every incoming is ONE value states
                            # the same equivalence as a single incoming: the
                            # port IS that value on every path (the
                            # return-merge field Phi of a span the function
                            # forwards unwritten from every return site --
                            # N=2 orbital dt system, ``pub_limits`` of
                            # ``step_with_dt_control_used``: Phi(f, f, f),
                            # read by a call in the retry loop, where the
                            # Phi's block does not dominate).  The CFG check
                            # below still decides.
                            replacement = defining.args[0]
                if replacement is None:
                    _note("no-candidate", tuple(
                        definition_sites.get(int(argument.id), ())
                    )[:3])
                    continue
                if replacement is argument:
                    continue
                incoming_block = (
                    str(incoming_blocks[argument_index])
                    if argument_index < len(incoming_blocks) else None
                )
                if dominates(
                    argument, block_name, instruction_index, incoming_block
                ):
                    _note("argument-dominates", (
                        int(argument.id) in argument_ids,
                        tuple(definition_sites.get(int(argument.id), ()))[:2],
                    ))
                    continue
                _note("argument-not-dominating", tuple(
                    definition_sites.get(int(argument.id), ())
                )[:3])
                if not dominates(
                    replacement, block_name, instruction_index, incoming_block
                ):
                    _note("candidate-not-dominating", (
                        int(replacement.id),
                        tuple(definition_sites.get(int(replacement.id), ()))[:3],
                    ), replacement)
                    continue
                _note("replaced", int(replacement.id), replacement)
                # The substitution itself is a fact on the book, one row per
                # occurrence: DERIVED(the original's and the replacement's
                # ``ssa_value`` cells, the blocks the dominance proof names)
                # with the proof's evidence in the fact.  Not swallowed: a
                # substitution the book did not take is not a valid one.
                from .ssa_record_return_state import post_dominance_rebinding

                use_target = (
                    str(incoming_block) if incoming_block is not None
                    else str(block_name)
                )
                replacement_sites = tuple(
                    definition_sites.get(int(replacement.id), ())
                )
                post_dominance_rebinding(
                    function, LOOP_RESULT_USE_REBINDING, (
                        str(function.metadata.get(
                            "tensor_shape_concordance_scope"
                        ) or function.name),
                        str(block_name), int(instruction_index),
                        int(argument_index),
                    ), (
                        int(argument.id), int(replacement.id),
                        (
                            ("formal",) if int(replacement.id) in argument_ids
                            else replacement_sites
                        ),
                        tuple(definition_sites.get(int(argument.id), ())),
                        use_target,
                        "phi_edge" if incoming_block is not None
                        else int(instruction_index),
                    ),
                    original=argument, replacement=replacement,
                    blocks=(
                        use_target,
                        *(site[0] for site in replacement_sites[:1]),
                    ),
                    stage=CONTROL_SSA_FINISH, mode=Mode.REVISE,
                )
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
    region_value_ranks: Mapping[int, int] | None = None,
    tensor_shape_concordance_scope: str | None = None,
    lexical_read_scope: Any = None,
    shape_scope: Any = None,
    plan_callsite_bindings: Mapping[
        int, tuple[tuple[int, ...], tuple[int, ...]]
    ] | None = None,
    value_aliases: Mapping[int, int] | None = None,
    inout_value_ids: tuple[int, ...] = (),
    constant_value_ids: tuple[int, ...] = (),
    output_value_ids: tuple[int, ...] = (),
    named_output_histories: Mapping[str, tuple[int, ...]] | None = None,
    value_name_histories: Mapping[str, tuple[int, ...]] | None = None,
    parameter_names: tuple[str, ...] = (),
    sequence_initializations: tuple[tuple[int, str, int], ...] = (),
    sequence_declarations: tuple[tuple[int, str, int, bool], ...] = (),
    sequence_column_dtypes: Mapping[int, tuple[str, ...]] | None = None,
    sequence_record_identities: Mapping[int, str] | None = None,
    sequence_row_record_slots: Mapping[
        int, tuple[tuple[int, str, int], ...]
    ] | None = None,
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
    nested_tensor_sequence_ids: tuple[int, ...] = (),
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
        region_value_ranks=region_value_ranks,
        tensor_shape_concordance_scope=tensor_shape_concordance_scope,
        lexical_read_scope=lexical_read_scope,
        shape_scope=shape_scope,
        plan_callsite_bindings=plan_callsite_bindings,
        value_aliases=value_aliases,
        inout_value_ids=inout_value_ids,
        constant_value_ids=constant_value_ids,
        output_value_ids=output_value_ids,
        named_output_histories=named_output_histories,
        value_name_histories=value_name_histories,
        parameter_names=parameter_names,
        sequence_initializations=sequence_initializations,
        sequence_declarations=sequence_declarations,
        sequence_column_dtypes=sequence_column_dtypes,
        sequence_record_identities=sequence_record_identities,
        sequence_row_record_slots=sequence_row_record_slots,
        source_sequence_ids=source_sequence_ids,
        sequence_memberships=sequence_memberships,
        table_lookups=table_lookups,
        lexical_table_lookup_result_ids=lexical_table_lookup_result_ids,
        table_lookup_defaults=table_lookup_defaults,
        table_stores=table_stores,
        table_deletions=table_deletions,
        retained_sequence_ids=retained_sequence_ids,
        nested_sequence_ids=nested_sequence_ids,
        nested_tensor_sequence_ids=nested_tensor_sequence_ids,
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
        }, owner=helper.name)
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

    from .identity_concordance import current_identity_book as _navigation_book

    class Builder:
        def __init__(self, name: str, dtypes: tuple[str, ...]):
            self.name = name
            # Every value of a navigation function is minted through the
            # book under the function's own scope: formals from the function
            # root, results from their operands' cells.
            self.book = _navigation_book()
            self.scope = str(name)
            self.root = _function_root_cell(self.book, self.scope)
            self.args = [
                SSAValue(self._mint(CONTROL_EXPRESSION, (), dtype), dtype=dtype)
                for dtype in dtypes
            ]
            self.instructions: list[Instr] = []

        def _mint(self, transform: Any, args: Any, dtype: str) -> int:
            operands = tuple(
                cell for cell in (
                    self.book.latest_ref(SSA_VALUE, (self.scope, int(value.id)))
                    for value in args
                )
                if cell is not None
            ) or (self.root,)
            return _mint_ssa_id(
                self.book, self.scope, transform, operands,
                dtype=dtype, stage=CONTROL_SSA_ENTRY,
            )

        def emit(self, operation: Handler, args=(), *, dtype="i32", **attributes):
            result = SSAValue(self._mint(
                CONTROL_CONST if operation is Handler.Const else CONTROL_EXPRESSION,
                args, dtype,
            ), dtype=dtype)
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
    separate_field_storage: bool = False,
    field_accounting: Mapping[int, Mapping[str, Any]] | None = None,
    field_write_state_cells: tuple[Any, ...] = (),
    lexical_read_scope: Any = None,
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

    A write whose reducer WRITTEN cell (``field_write_state_cells``, one per
    ``"write"`` op) was lowered at its authored site -- control lowering
    emitted the ``scalar_record_field_assignment`` Store in the arm that owns
    the assignment and published that arm's value under the cell on
    ``ssa_field_version`` -- is OWNED by that arm: this Store is rewritten in
    place into the real slot store, and no second store is placed after the
    source's producer (which, for a literal or a value computed before the
    ``if``, is a dominating block and would run on every path).  The arm is
    the ``control_block_placement`` row of the cell's ``scalar_field_write``
    block; the slot address derives from that row.
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
    # Field-slot cells are FIELD_SLOT_ACCESS mints on the control function's
    # scope, from the receiver's ``ssa_value`` cell (its parameter seed) or
    # the function root when the receiver has no row.
    from .identity_concordance import current_identity_book as _slot_book

    _slot_scope = str(
        control_function.metadata.get("tensor_shape_concordance_scope")
        or control_function.name
    )
    _receiver_cell = _slot_book().latest_ref(
        SSA_VALUE, (_slot_scope, int(self_value_id)),
    )

    def fresh(*causes: Any) -> int:
        book = _slot_book()
        operand = (
            _receiver_cell if _receiver_cell is not None
            else _function_root_cell(book, _slot_scope)
        )
        return _mint_ssa_id(
            book, _slot_scope, FIELD_SLOT_ACCESS, (operand, *causes),
            dtype=dtype, stage=CONTROL_SSA_FINISH,
        )

    # A receiver is physically a set of typed columns.  The traditional one
    # arena form is the degenerate (and fastest) one-column case.  Slots keep a
    # logical index, while this table resolves that index to (arena, offset),
    # so mixed fields never need coercion or a tagged-object runtime.
    slot_dtypes = {
        int(slot): str((field_dtypes or {}).get(int(slot), dtype))
        for _kind, _value_id, slot in field_ops
    }
    column_keys = (
        tuple((slot_dtypes[slot], int(slot)) for slot in sorted(slot_dtypes))
        if separate_field_storage else
        tuple((column_dtype, None) for column_dtype in sorted(
            set(slot_dtypes.values()) or {str(dtype)}
        ))
    )
    slots_by_column = {
        key: (
            (key[1],) if key[1] is not None else tuple(sorted(
                slot for slot, slot_dtype in slot_dtypes.items()
                if slot_dtype == key[0]
            ))
        )
        for key in column_keys
    }
    column_arrays: dict[tuple[str, int | None], SSAValue] = {}
    for index, key in enumerate(column_keys):
        column_dtype = key[0]
        arena_id = int(self_value_id) if index == 0 else fresh()
        column_arrays[key] = SSAValue(
            arena_id, dtype=column_dtype,
            shape=(len(slots_by_column[key]),),
            accounting=(
                dict((field_accounting or {}).get(int(key[1]), {}))
                if key[1] is not None else {}
            ),
        )
    field_locations = {
        slot: (
            int(column_arrays[(slot_dtype, slot if separate_field_storage else None)].id),
            slots_by_column[(slot_dtype, slot if separate_field_storage else None)].index(slot),
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

    def slot_address(
        slot: int, *causes: Any,
    ) -> tuple[list[Instr], SSAValue, str]:
        arena_id, offset, slot_dtype = field_locations[int(slot)]
        index = SSAValue(fresh(*causes), dtype="int64")
        address = SSAValue(fresh(*causes))
        return (
            [
                Instr("Const", [], index, attributes={"value": int(offset)}),
                Instr("GetElementPtr", [column_arrays[(
                    slot_dtype, int(slot) if separate_field_storage else None
                )], index], address),
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
    # Arm-owned writes.  Control lowering emitted each declared scalar field
    # assignment as a ``scalar_record_field_assignment`` Store at the block's
    # own site (inside the arm that owns it) and published its value on
    # ``ssa_field_version`` under the write's reducer cell; the write op of
    # ``field_ops`` names that same cell.  The authored site is therefore a
    # lookup by cell, not a search for the source's producer.
    arm_stores: dict[Any, int] = {}
    for position, (_name, instruction) in enumerate(flat):
        if (
            instruction.op in {"Store", "store"}
            and (instruction.attributes or {}).get("binding")
            == "scalar_record_field_assignment"
            and isinstance(
                (instruction.attributes or {}).get("field_state_cell"), Ref
            )
        ):
            arm_stores.setdefault(
                instruction.attributes["field_state_cell"], position
            )
    _write_cells = (
        iter(field_write_state_cells)
        if len(field_write_state_cells)
        == sum(1 for op in field_ops if op[0] == "write") else iter(())
    )
    _read_ids = {
        int(value_id) for kind, value_id, _slot in field_ops if kind == "read"
    }
    # op index -> flat position of the arm Store that IS that write; and the
    # positions whose destination is a field READ (the placeholder named the
    # field by its read value, which injection turns into a Load -- data, not
    # an address, so that Store does nothing and is replaced by the real one).
    arm_owned: dict[int, tuple[int, Any]] = {}
    placeholder_positions: set[int] = set()
    claimed_positions: set[int] = set()
    for op_index, (kind, value_id, _slot) in enumerate(field_ops):
        if kind != "write":
            continue
        cell = next(_write_cells, None)
        position = arm_stores.get(cell)
        if position is None or position in claimed_positions:
            continue
        claimed_positions.add(position)
        arm_owned[op_index] = (position, cell)
        if int(flat[position][1].args[1].id) in _read_ids:
            placeholder_positions.add(position)
    replaced_at: dict[int, list[Instr]] = {}

    producer_position: dict[int, int] = {}
    produced_values: dict[int, SSAValue] = {}
    first_consumer_position: dict[int, int] = {}
    for position, (_name, instruction) in enumerate(flat):
        if instruction.res is not None:
            producer_position.setdefault(int(instruction.res.id), position)
            produced_values.setdefault(int(instruction.res.id), instruction.res)
        for index, argument in enumerate(instruction.args):
            if index == 1 and position in placeholder_positions:
                continue  # the replaced placeholder's destination: not a use
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
    def arm_placement(cell: Any) -> Any:
        """The ``control_block_placement`` cell of the ``scalar_field_write``
        block that IS ``cell`` (None when the book has no such row)."""

        if lexical_read_scope is None:
            return None
        from .concordance_declarations import (
            CONTROL_BLOCK, CONTROL_BLOCK_PLACEMENT, ControlBlockKind,
        )

        book = _slot_book()
        scope = tuple(lexical_read_scope)
        block = book.latest_ref(
            CONTROL_BLOCK, (scope, ControlBlockKind.SCALAR_FIELD_WRITE, cell),
        )
        return None if block is None else book.latest_ref(
            CONTROL_BLOCK_PLACEMENT, (scope, block),
        )

    # The reads this injection materialises as slot Loads: each is a VERSION
    # of its field, read at its own place and time, not an address of it.  It
    # is posted on ``receiver_field_slot_read`` so record-ABI materialization
    # reads the decision from the book.
    from .concordance_declarations import (
        RECEIVER_FIELD_SLOT_READ, ReceiverFieldSlotReadFact,
    )

    _write_ops = [op for op in field_ops if op[0] == "write"]
    _slot_write_cell: dict[int, Any] = {}
    if len(field_write_state_cells) == len(_write_ops):
        for (_k, _v, _slot), _cell in zip(_write_ops, field_write_state_cells):
            if isinstance(_cell, Ref):
                _slot_write_cell.setdefault(int(_slot), _cell)

    def _post_slot_read(read_id: int, slot: int, address: SSAValue) -> None:
        book = _slot_book()
        sources = tuple(dict.fromkeys(
            cell for cell in (
                book.latest_ref(SSA_VALUE, (_slot_scope, int(address.id))),
                book.latest_ref(SSA_VALUE, (_slot_scope, int(read_id))),
                _slot_write_cell.get(int(slot)),
            ) if isinstance(cell, Ref)
        ))
        if not sources:
            return
        book.post(
            RECEIVER_FIELD_SLOT_READ, (_slot_scope, int(read_id)),
            ReceiverFieldSlotReadFact(int(slot), int(read_id)),
            stage=CONTROL_SSA_FINISH, provenance=Derived(sources),
            mode=Mode.CONCORD,
        )

    materialized_references: set[int] = set()
    for schedule_index, (kind, value_id, slot) in enumerate(field_ops):
        if schedule_index in arm_owned:
            position, write_cell = arm_owned[schedule_index]
            authored = flat[position][1]
            source = authored.args[0]
            placement = arm_placement(write_cell)
            causes = () if placement is None else (placement,)
            prelude, address, slot_dtype = slot_address(slot, *causes)
            group = []
            # A static reference has no producer in the control body: it is
            # materialised once where it dominates every use (the entry --
            # a handle constant, no effect), as a reference write always was.
            # Only the STORE of it is the arm's.
            reference = reference_sources.get(int(source.id))
            if reference is not None and int(source.id) not in (
                producer_position
            ) and int(source.id) not in materialized_references:
                materialized_references.add(int(source.id))
                reference_value = SSAValue(int(source.id), dtype="opaque_ref")
                insertions.append((0, schedule_index, entry_name, [Instr(
                    Handler.StaticRef.value, [], reference_value,
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
                )]))
            # A literal is materialised in the slot's dtype, as a constant
            # field write always was.
            literal = producer_position.get(int(source.id))
            if literal is not None and flat[literal][1].op == "Const" and (
                not flat[literal][1].args
            ):
                source = SSAValue(fresh(*causes), dtype=slot_dtype)
                group.append(Instr(
                    "Const", [], source,
                    attributes=dict(flat[literal][1].attributes or {}),
                ))
            group += [*prelude, Instr(
                "Store", [source, address], None,
                attributes={
                    "opaque_reference_storage": True, "field_slot": int(slot),
                } if slot_dtype == "opaque_ref" else {},
            )]
            # A placeholder that named the field by a read is replaced; one
            # that names the field's incoming formal stays (it writes that
            # formal in this arm) and the slot store follows it.
            replaced_at[position] = (
                group if position in placeholder_positions
                else [authored, *group]
            )
            continue
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
            _post_slot_read(int(value_id), int(slot), address)
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
        if position in replaced_at:
            rebuilt[name].extend(replaced_at[position])
            continue
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
    arguments = [column_arrays[key] for key in column_keys]
    seen_argument_ids = {int(argument.id) for argument in arguments}
    for argument in control_function.args:
        argument_id = int(argument.id)
        if (
            argument_id in seen_argument_ids
            or argument_id in field_read_ids
            or argument_id in materialized_references
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
    callsite_projection_ids: Mapping[int, tuple[int, ...]] | None = None,
    pure_region_indices: frozenset[int] = frozenset(),
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
        elif isinstance(block, StateMachineTick):
            for _value, body in block.cases:
                discover_control(body)
            if block.default is not None:
                discover_control(block.default)

    discover_control(control.root)
    # Callsites lexically owned by a conditional arm (see
    # ConditionalBlock.body_callsite_ids): scheduled inside that arm by
    # `rebuild`, never anchored on a later region or left trailing.
    arm_callsite_candidates: set[int] = set()

    def discover_arm_callsites(block: Any) -> None:
        if isinstance(block, SequenceBlock):
            for child in block.blocks:
                discover_arm_callsites(child)
        elif isinstance(block, ConditionalBlock):
            arm_callsite_candidates.update(map(int, block.body_callsite_ids))
            arm_callsite_candidates.update(map(int, block.orelse_callsite_ids))
            discover_arm_callsites(block.body)
            if block.orelse is not None:
                discover_arm_callsites(block.orelse)
        elif isinstance(block, (LoopBlock, WhileBlock)):
            discover_arm_callsites(block.body)
        elif isinstance(block, CallBlock):
            discover_arm_callsites(block.callee)
        elif isinstance(block, StateMachineTick):
            for _value, body in block.cases:
                discover_arm_callsites(body)
            if block.default is not None:
                discover_arm_callsites(block.default)

    discover_arm_callsites(control.root)
    # Callsites the shell already placed at their authored position (a
    # ``__plan_callsite_N__`` statement in the tree) are not scheduled
    # again: authored placement wins over "before the next retained region".
    placed_callsites: set[int] = set()

    def discover_placed(block: Any) -> None:
        if isinstance(block, StatementBlock):
            for line in block.lines:
                match = _CALLSITE_MARKER.fullmatch(str(line))
                if match is not None:
                    placed_callsites.add(int(match.group(1)))
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                discover_placed(child)
        elif isinstance(block, ConditionalBlock):
            discover_placed(block.body)
            if block.orelse is not None:
                discover_placed(block.orelse)
        elif isinstance(block, (LoopBlock, WhileBlock)):
            if isinstance(block, WhileBlock):
                discover_placed(block.condition)
            discover_placed(block.body)
        elif isinstance(block, CallBlock):
            discover_placed(block.callee)
        elif isinstance(block, StateMachineTick):
            # A tick's arms hold the calls its dispatch selects; a marker in
            # an arm is placed there, not "before the next region".
            for _value, body in block.cases:
                discover_placed(body)
            if block.default is not None:
                discover_placed(block.default)

    discover_placed(control.root)
    arm_owned_callsites: set[int] = set()
    calls_before: dict[int, list[int]] = {}
    replaced_projection_regions: set[int] = set()
    pending: list[int] = []
    last_call: PlanCall | None = None

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
            last_call = item
            if int(item.callsite_id) in placed_callsites:
                continue
            if int(item.callsite_id) in arm_callsite_candidates:
                arm_owned_callsites.add(int(item.callsite_id))
                continue
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
            pending = []
        if (
            region in retained_regions
            and last_call is not None
            and is_call_projection_region(item, last_call)
        ):
            # A region that only unpacks a call's bound results is the
            # call's projection, whether the call was placed lexically or
            # is scheduled here; its work is the call itself.
            replaced_projection_regions.add(region)
            calls_before.setdefault(region, [])
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

    def append_before_terminal(body: Any, trailing_marker: Any) -> SequenceBlock:
        """Append a trailing call marker ahead of the body's terminal edge.

        A loop body that ends in an unconditional ``break``/``continue``/
        ``return`` has no reachable position after that edge; a marker
        appended there is dead code and the call it stands for never runs.
        """

        blocks = list(sequence(body).blocks)
        index = len(blocks)
        while (
            index > 0
            and isinstance(blocks[index - 1], LoopControlBlock)
            and blocks[index - 1].predicate_value_id is None
            and blocks[index - 1].predicate_expression is None
        ):
            index -= 1
        blocks.insert(index, trailing_marker)
        return SequenceBlock(tuple(blocks))

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
                rebuilt_body = append_before_terminal(
                    rebuilt_body, marker(trailing)
                )
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
                rebuilt_body = append_before_terminal(
                    rebuilt_body, marker(trailing)
                )
            return replace(
                block, condition=rebuilt_condition, body=rebuilt_body
            )
        if isinstance(block, ConditionalBlock):
            # Planned calls whose only lexical home is this arm (no region
            # of their own to anchor on) are scheduled at the head of the
            # arm, ahead of its regions and its return control.
            owned_body = tuple(
                callsite_id for callsite_id in block.body_callsite_ids
                if callsite_id in arm_owned_callsites
            )
            owned_else = tuple(
                callsite_id for callsite_id in block.orelse_callsite_ids
                if callsite_id in arm_owned_callsites
            )
            rebuilt_body = rebuild(block.body)
            if owned_body:
                rebuilt_body = sequence(marker(owned_body), rebuilt_body)
            rebuilt_else = (
                rebuild(block.orelse) if block.orelse is not None else None
            )
            if owned_else:
                rebuilt_else = sequence(
                    marker(owned_else),
                    rebuilt_else if rebuilt_else is not None
                    else SequenceBlock(()),
                )
            return replace(block, body=rebuilt_body, orelse=rebuilt_else)
        if isinstance(block, StateMachineTick):
            return replace(
                block,
                cases=tuple(
                    (value, rebuild(body)) for value, body in block.cases
                ),
                default=(
                    None if block.default is None else rebuild(block.default)
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
        elif isinstance(block, StateMachineTick):
            for _value, body in block.cases:
                produced.update(produced_sequences(body))
            if block.default is not None:
                produced.update(produced_sequences(block.default))
        return produced


    def expression_value_ids(expression: Any) -> set[int]:
        """Every value id a control expression reads, transitively.

        A predicate such as ``not ok`` is a control expression over the
        call result ``ok``; the block's own predicate id is the negation
        node, which nothing publishes.  The dependency is on the operands.
        """

        found: set[int] = set()
        pending = [expression]
        while pending:
            current = pending.pop()
            if current is None:
                continue
            if getattr(current, "value_id", None) is not None:
                found.add(int(current.value_id))
            pending.extend(getattr(current, "operands", ()) or ())
        return found

    def dependency_signature(
        block: Any,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Return exact straight-line inputs and publications for a block."""

        if isinstance(block, LoopControlBlock):
            # A tuple return may be the sole consumer of a trailing source
            # call. Keep its value dependencies visible so that call is
            # scheduled before the function exit, not in unreachable code.
            inputs = {int(value_id) for value_id in block.return_value_ids
                      if value_id is not None}
            if block.predicate_value_id is not None:
                inputs.add(int(block.predicate_value_id))
            inputs |= expression_value_ids(block.predicate_expression)
            return tuple(sorted(inputs)), ()
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
            # A call publishes its bound results and every projection the
            # caller takes of them (``metrics.mass_err``, ``result[1]``):
            # those projections are what regions, loops and effects consume.
            # Without them a loop reading the record's fields carried no
            # dependency on the call and floated ahead of it.
            return (
                tuple(dict.fromkeys(
                    value_id
                    for callsite_id in callsites
                    for value_id in bindings.get(callsite_id, ((), ()))[0]
                )),
                tuple(dict.fromkeys((
                    *(
                        value_id
                        for callsite_id in callsites
                        for value_id in bindings.get(callsite_id, ((), ()))[1]
                    ),
                    *(
                        int(value_id)
                        for callsite_id in callsites
                        for value_id in (callsite_projection_ids or {}).get(
                            int(callsite_id), ()
                        )
                    ),
                ))),
            )
        if isinstance(block, SequenceMutationBlock):
            mutation = block.mutation
            consumed_ids: set[int] = set(map(int, mutation.argument_value_ids))
            if mutation.operator == "replace":
                # Copy reads the source arena at this lexical point. Its id
                # is not an SSA result produced by a future append or clear.
                consumed_ids.difference_update(map(int, mutation.argument_value_ids))
            consumed_ids |= expression_value_ids(mutation.predicate_expression)
            for argument_expression in mutation.argument_expressions:
                consumed_ids |= expression_value_ids(argument_expression)
            return (
                tuple(sorted(consumed_ids)),
                (
                    int(mutation.effect_node_id),
                    int(mutation.sequence_value_id),
                ),
            )
        if isinstance(block, ScalarFieldWriteBlock):
            return (tuple(sorted(expression_value_ids(block.value_expression))),
                    (int(block.effect_node_id),))
        if isinstance(block, SequenceQueryBlock):
            return (
                tuple(dict.fromkeys((
                    # A resident arena is storage, not a value defined by
                    # whichever mutation next mentions its id. In particular
                    # the first read must not depend on a later clear. A
                    # generator query has a separate, explicit producer.
                    *((int(block.sequence_value_id),)
                      if block.producer_loop_node_id is not None else ()),
                    *(
                        () if block.default_value_id is None
                        else (int(block.default_value_id),)
                    ),
                    *map(int, block.reduction_prefix_value_ids),
                    *map(int, block.reduction_suffix_value_ids),
                ))),
                tuple(dict.fromkeys((
                    int(block.result_value_id),
                    *map(int, block.result_alias_ids),
                ))),
            )
        if isinstance(block, ConditionalBlock):
            # Keep the branch atomic, but expose its external dependencies.
            # Otherwise an overlaid region used as a call anchor can strand
            # that call after a branch which already consumes its result.
            consumed: set[int] = {int(block.predicate_value_id)}
            consumed |= expression_value_ids(block.predicate_expression)
            produced: set[int] = set()

            def collect_arm(candidate: Any) -> bool:
                if isinstance(candidate, SequenceBlock):
                    return all(collect_arm(child) for child in candidate.blocks)
                signature = dependency_signature(candidate)
                if signature is None:
                    return False
                consumed.update(map(int, signature[0]))
                produced.update(map(int, signature[1]))
                return True

            if not collect_arm(block.body) or (
                block.orelse is not None and not collect_arm(block.orelse)
            ):
                return None
            # Arm-local values do not become unconditional publications.
            # Only the control IR's explicit merges cross this boundary.
            aliases = (*block.carried_aliases, *block.carried_sequence_aliases)
            consumed.update(int(value_id)
                            for yes, no, _result in block.result_aliases
                            for value_id in (yes, no))
            external = consumed - produced
            external.add(int(block.predicate_value_id))
            external |= expression_value_ids(block.predicate_expression)
            external.update(int(initial) for _yes, _no, initial, _merged in aliases)
            return (
                tuple(sorted(external)),
                tuple(sorted({
                    *(int(merged) for _yes, _no, _initial, merged in aliases),
                    *(int(merged) for _yes, _no, merged in block.result_aliases),
                })),
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
            if region_index in pure_region_indices:
                # Pure arithmetic already has complete SSA dependencies.
                # Its old flat rank is not an effect-order constraint: keeping
                # it can force an early rejection predicate after accept work.
                return None
            return hierarchy_statement_rank.get(("region", region_index))
        callsites = []
        for line in block.lines:
            match = _CALLSITE_MARKER.fullmatch(str(line))
            if match is None:
                return None
            callsites.append(int(match.group(1)))
        # A source-placed call already has an execution position in Control
        # IR. The flat hierarchy can put an effect-only call after a read
        # because no result dependency connects them; that rank must not
        # override the authored placement during the stable sort below.
        positions = tuple(
            hierarchy_statement_rank[("call", callsite_id)]
            for callsite_id in callsites
            if ("call", callsite_id) in hierarchy_statement_rank
            and callsite_id not in placed_callsites
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
                dependency_reasons: dict[tuple[int, int], list[Any]] = {}

                def require(
                    current: int, prerequisite: int, reason: Any,
                ) -> None:
                    dependencies[current].add(prerequisite)
                    dependency_reasons.setdefault(
                        (int(prerequisite), int(current)), []
                    ).append(reason)
                for position, signature in enumerate(signatures):
                    assert signature is not None
                    required: set[int] = set()
                    for feed in signature[0]:
                        candidates = producers.get(int(feed), ())
                        prior = [candidate for candidate in candidates if candidate < position]
                        later = [candidate for candidate in candidates if candidate > position]
                        if prior:
                            prerequisite = max(prior)
                            required.add(prerequisite)
                            dependency_reasons.setdefault(
                                (int(prerequisite), int(position)), []
                            ).append(("value", int(feed)))
                        elif later:
                            prerequisite = min(later)
                            required.add(prerequisite)
                            dependency_reasons.setdefault(
                                (int(prerequisite), int(position)), []
                            ).append(("value", int(feed)))
                    dependencies.append(required)
                hierarchy_positions = [
                    hierarchy_statement_position(child) for child in run
                ]
                # The hierarchy is a flat data/effect plan; a lexical terminal
                # is a control boundary absent from that flat ordering. Chaining
                # ranked statements across one can order fallthrough work before
                # the guarded return/break/continue that precedes it. Combined
                # with the terminal guard below, that manufactures a cycle even
                # though the two orders describe mutually exclusive paths.
                # Preserve hierarchy precedence independently on each side.
                terminal_segments: list[int] = []
                terminal_segment = 0
                for child in run:
                    terminal_segments.append(terminal_segment)
                    if isinstance(child, LoopControlBlock):
                        terminal_segment += 1
                ranked_by_segment: dict[int, list[tuple[int, int]]] = {}
                for position, hierarchy_position in enumerate(hierarchy_positions):
                    if hierarchy_position is None:
                        continue
                    ranked_by_segment.setdefault(
                        terminal_segments[position], []
                    ).append((int(hierarchy_position), position))
                for ranked in ranked_by_segment.values():
                    ranked.sort()
                    for (_prior_rank, prior), (_rank, current) in zip(
                        ranked, ranked[1:]
                    ):
                        require(current, prior, (
                            "hierarchy", int(_prior_rank), int(_rank)
                        ))

                def sequence_accesses(candidate):
                    """One arena's reads and its writes, told apart.

                    These used to be one set, and every pair of accesses to an
                    arena was then ordered -- including READ AFTER READ, which
                    needs no ordering at all because two reads of the same
                    sequence commute.  Those spurious edges are not merely
                    conservative: chained with a real dependency they close
                    cycles that do not exist, and the scheduler then refuses a
                    whole program with "control effect order conflicts with
                    value dependencies".  One such refusal chained three pure
                    readers of one arena into a path from a placed callsite to
                    the loop's own control block, which the terminal-guard rule
                    below independently orders the other way.

                    A mutation writes its sequence; a ``replace`` also READS the
                    values it writes in; a query reads.
                    """
                    reads, writes = set(), set()
                    if isinstance(candidate, SequenceMutationBlock):
                        writes.add(int(candidate.mutation.sequence_value_id))
                        if candidate.mutation.operator == "replace":
                            reads.update(map(int, candidate.mutation.argument_value_ids))
                    elif isinstance(candidate, SequenceQueryBlock):
                        if candidate.producer_loop_node_id is None:
                            reads.add(int(candidate.sequence_value_id))
                    for mutation in getattr(candidate, 'sequence_mutations', ()):
                        writes.add(int(mutation.sequence_value_id))
                    for child in getattr(candidate, 'blocks', ()):
                        child_reads, child_writes = sequence_accesses(child)
                        reads |= child_reads
                        writes |= child_writes
                    for field in ('condition', 'body', 'orelse', 'callee'):
                        child = getattr(candidate, field, None)
                        if child is not None:
                            child_reads, child_writes = sequence_accesses(child)
                            reads |= child_reads
                            writes |= child_writes
                    return reads, writes

                # Three of the four dependence kinds, and not the fourth:
                #   RAW  a read after a write   -- last_write
                #   WAR  a write after a read   -- last_access
                #   WAW  a write after a write  -- last_access
                #   RAR  a read after a read    -- NONE, reads commute
                last_access: dict[int, int] = {}
                last_write: dict[int, int] = {}
                last_terminal = None
                for position, child in enumerate(run):
                    reads, writes = sequence_accesses(child)
                    for arena in reads:
                        if arena in last_write:
                            require(position, last_write[arena], (
                                "sequence_raw", int(arena)
                            ))
                    for arena in writes:
                        if arena in last_access:
                            access_kind = (
                                "sequence_waw"
                                if arena in last_write
                                and last_write[arena] == last_access[arena]
                                else "sequence_war"
                            )
                            require(position, last_access[arena], (
                                access_kind, int(arena)
                            ))
                    for arena in reads | writes:
                        last_access[arena] = position
                    for arena in writes:
                        last_write[arena] = position
                    # A source-placed call following a guarded continue/return
                    # must not execute on that terminal edge while waiting for
                    # the guard's numerical producer.
                    if isinstance(child, LoopControlBlock):
                        last_terminal = position
                    elif isinstance(child, StatementBlock) and last_terminal is not None:
                        if any(
                            (match := _CALLSITE_MARKER.fullmatch(str(line)))
                            and int(match.group(1)) in placed_callsites
                            for line in child.lines
                        ):
                            require(position, last_terminal, ("terminal_guard",))

                placed, active = set(), set()

                def schedule(position):
                    if position in placed:
                        return
                    if position in active:
                        def label(index):
                            child = run[index]
                            if isinstance(child, StatementBlock):
                                call_ids = tuple(
                                    int(match.group(1))
                                    for line in child.lines
                                    if (match := _CALLSITE_MARKER.fullmatch(str(line)))
                                )
                                if call_ids and len(call_ids) == len(child.lines):
                                    return tuple(
                                        (
                                            "PlanCall", callsite_id,
                                            planned_calls[callsite_id].callee.name
                                            if callsite_id in planned_calls else None,
                                        )
                                        for callsite_id in call_ids
                                    )
                            return (
                                child.lines if isinstance(child, StatementBlock)
                                else (type(child).__name__,
                                      getattr(child, 'source_node_id', None),
                                      getattr(child, 'source_call_node_id', None),
                                      getattr(child, 'predicate_value_id', None),
                                      getattr(child, 'action', None),
                                      getattr(child, 'site_node_id', None))
                            )
                        raise ValueError(
                            'control effect order conflicts with value dependencies: '
                            f'cycle={[(i, label(i), sorted(dependencies[i]), signatures[i]) for i in sorted(active)]}; '
                            f'edges={[(prerequisite, current, tuple(reasons)) for (prerequisite, current), reasons in sorted(dependency_reasons.items()) if prerequisite in active and current in active]}'
                        )
                    active.add(position)
                    for dependency in sorted(dependencies[position]):
                        schedule(dependency)
                    active.remove(position)
                    placed.add(position)
                    ordered.append(run[position])

                # Pull prerequisites immediately before their first consumer.
                # Choosing any ready node instead could postpone a conditional
                # append past a later truth read, or run accept calls before a
                # rejection guard whose predicate was not available yet.
                for position in range(len(run)):
                    schedule(position)
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
        if isinstance(block, StateMachineTick):
            return replace(
                block,
                cases=tuple(
                    (value, dependency_order(body))
                    for value, body in block.cases
                ),
                default=(
                    None if block.default is None
                    else dependency_order(block.default)
                ),
            )
        return block

    return replace(
        rebuilt_control,
        root=dependency_order(rebuilt_control.root),
    ), bindings


def debug_region_output_loads(function: Function, tag: str) -> None:
    """Env-gated probe: report region-output loads whose result id drifted.

    ``TURING_DEBUG_OUTPUT_LOADS=<function-name-substring>`` prints, at each
    call site that invokes this, every ``Load`` stamped with a
    ``source_output_id`` whose result id no longer equals that stamp.
    """

    wanted = os.environ.get("TURING_DEBUG_OUTPUT_LOADS")
    if not wanted or wanted not in str(function.name):
        return
    drifted = [
        (block_name, int(instruction.attributes.get("source_output_id")),
         int(instruction.res.id))
        for block_name, block in function.blocks.items()
        for instruction in block.instrs
        if instruction.res is not None
        and instruction.attributes
        and instruction.attributes.get("source_output_id") is not None
        and int(instruction.attributes.get("source_output_id"))
        != int(instruction.res.id)
    ]
    print(
        f"DEBUG-OUTPUT-LOADS [{tag}] {function.name}: drifted={drifted}",
        file=sys.stderr, flush=True,
    )
    if tag != os.environ.get("TURING_DEBUG_OUTPUT_LOADS_TRACE_AT", "shell-after-sections"):
        return
    import traceback as _traceback

    def _where() -> str:
        frames = _traceback.extract_stack(limit=9)[:-2]
        return " <- ".join(f"{f.name}:{f.lineno}" for f in reversed(frames))

    class _TracedValue(SSAValue):
        @property
        def id(self):  # type: ignore[override]
            return self.__dict__["id"]

        @id.setter
        def id(self, new_id):
            old_id = self.__dict__.get("id")
            if old_id is not None and int(old_id) != int(new_id):
                print(f"DEBUG-OUTPUT-LOADS-TRACE {function.name}: value id "
                      f"{old_id} -> {new_id} at {_where()}", file=sys.stderr, flush=True)
            self.__dict__["id"] = new_id

    class _TracedInstr(Instr):
        @property
        def res(self):  # type: ignore[override]
            return self.__dict__["res"]

        @res.setter
        def res(self, value):
            old = self.__dict__.get("res")
            if old is not None and value is not None and int(old.id) != int(value.id):
                print(f"DEBUG-OUTPUT-LOADS-TRACE {function.name}: res object "
                      f"{int(old.id)} -> {int(value.id)} (source_output_id="
                      f"{(self.attributes or {}).get('source_output_id')}) at {_where()}",
                      file=sys.stderr, flush=True)
            self.__dict__["res"] = value

    for block in function.blocks.values():
        for instruction in block.instrs:
            if (
                instruction.res is not None
                and str(instruction.op) == "Load"
                and instruction.attributes
                and instruction.attributes.get("source_output_id") is not None
            ):
                instruction.res.__class__ = _TracedValue
                instruction.__class__ = _TracedInstr


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
    # Call-frame legalization runs after control lowering and can expose a
    # literal identity again after its first materialization was dead-code
    # eliminated.  Preserve the exact specialized source ledger so that late
    # recovery never has to guess through a transformed ProcessGraph.
    function.metadata["authored_constant_values"] = tuple(
        (int(value_id), copy.deepcopy(literal))
        for value_id, literal in sorted(constant_values.items())
    )
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
                # A proven dtype wins; otherwise the authored literal's own
                # Python type is exact.  ``return True`` had no proven dtype
                # and was declared float64 here, which only a later pass's
                # re-inference from the literal used to hide.
                dtype=(
                    "none" if is_none
                    else str(
                        value_dtypes.get(value_id)
                        or (
                            "bool" if isinstance(literal, bool)
                            else "int64" if isinstance(literal, int)
                            else "float64"
                        )
                    )
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


def _concord_region_feed_consumers(
    lexical_read_scope: Any,
    control_scope: Any,
    region_index: int,
    instructions: Any,
    feeds: Any,
) -> None:
    """Commit, per region feed, the graph consumers inside the region.

    Page ``region_feed_consumer`` row ``(control scope, region index, feed
    id)`` holds the region instructions' result ids (their graph consumer
    nodes) that read the feed.  Structure only: which binding each of those
    operands read is the base fact ``lexical_read_binding``.
    """

    if lexical_read_scope is None:
        return
    from .identity_concordance import current_identity_book

    book = current_identity_book()
    feed_ids = set(map(int, feeds))
    consumers: dict[int, list[int]] = {}
    for instruction in instructions:
        if instruction.res is None:
            continue
        for argument in instruction.args:
            if int(argument.id) in feed_ids:
                consumers.setdefault(int(argument.id), []).append(
                    int(instruction.res.id)
                )
    scope = tuple(lexical_read_scope)
    for feed, nodes in consumers.items():
        # DERIVED(the ``consumer_operand`` cells when the planner declared
        # them, else the consumers' and the feed's canonical cells).
        sources = []
        for node in dict.fromkeys(nodes):
            for cell in (
                (
                    book.latest_ref(
                        book.registry.pages["consumer_operand"],
                        (scope, int(node), int(feed)),
                    )
                    if "consumer_operand" in book.registry.pages else None
                ),
                book.latest_ref(CANONICAL_VALUE, (scope, int(node))),
            ):
                if cell is not None and cell not in sources:
                    sources.append(cell)
        feed_cell = book.latest_ref(CANONICAL_VALUE, (scope, int(feed)))
        if feed_cell is not None and feed_cell not in sources:
            sources.append(feed_cell)
        _post_derived_or_raw(
            book, REGION_FEED_CONSUMER,
            (str(control_scope), int(region_index), feed),
            tuple(dict.fromkeys(nodes)), tuple(sources),
            stage=CONTROL_SSA_REGION, mode=Mode.CONCORD,
        )


def _split_region_captures_by_binding(
    lexical_read_scope: Any,
    control_scope: Any,
    region_index: int,
    instructions: Any,
    captures: tuple[int, ...],
) -> tuple[int, ...]:
    """Give each binding a region reads through one captured value its formal.

    ``second = value`` then ``second = second + value`` inside a loop is one
    captured value read as two bindings; only ``second`` is loop-carried, so
    the two operands see different values at the call.  A region formal per
    value id cannot carry both.  When the book attributes a capture's
    operand slots (``lexical_read_binding`` at ``(read scope, instruction
    result, role, ordinal)``) to more than one binding and a loop carries
    that value under one of them (``loop_carried_binding``), every binding
    after the first gets a fresh formal and its slots are rewired to it.

    Page ``region_capture_binding`` row ``(control scope, region, formal)``
    -> ``(source value id, binding)`` records each split formal, the first
    binding's included; the region feed reads it by that binding.  Returns
    the captures with the new formals appended after their source.
    """

    if lexical_read_scope is None:
        return captures
    from ..common.tensors.topological_reducer import _operand_positions
    from .identity_concordance import current_identity_book

    book = current_identity_book()
    read_page = book.page("lexical_read_binding")
    carried_page = book.page("loop_carried_binding")
    carried_initials: dict[int, set[str]] = {}
    for row in carried_page.scope_rows(tuple(lexical_read_scope)):
        if len(row) == 4:
            carried_initials.setdefault(int(row[3]), set()).update(
                map(str, carried_page.latest(row) or ())
            )
    wanted = {int(value_id) for value_id in captures} & set(carried_initials)
    if not wanted:
        return captures
    slots: dict[int, list[tuple[Any, int, Any]]] = {}
    for instruction in instructions:
        if instruction.res is None:
            continue
        roles = list(getattr(instruction, "arg_roles", ()) or ())
        if len(roles) != len(instruction.args):
            continue
        for index, (role, ordinal, argument) in enumerate(_operand_positions(
            zip(instruction.args, roles)
        )):
            if int(argument.id) not in wanted:
                continue
            binding = read_page.latest((
                tuple(lexical_read_scope), int(instruction.res.id),
                role, ordinal,
            ))
            slots.setdefault(int(argument.id), []).append(
                (instruction, index, binding)
            )
    expanded: list[int] = []
    scope = tuple(lexical_read_scope)
    for value_id in map(int, captures):
        expanded.append(value_id)
        reads = slots.get(value_id)
        if not reads:
            continue
        bindings = list(dict.fromkeys(binding for *_slot, binding in reads))
        if (
            len(bindings) < 2
            or None in bindings
            or not set(bindings) & carried_initials[value_id]
        ):
            continue
        source_cell = book.latest_ref(CANONICAL_VALUE, (scope, value_id))

        def read_cells(binding: Any) -> tuple:
            """The ``lexical_read_binding`` cells of the slots that read
            ``binding``."""

            found = []
            for instruction, index, read in reads:
                if read != binding:
                    continue
                roles = list(getattr(instruction, "arg_roles", ()) or ())
                positions = list(_operand_positions(
                    zip(instruction.args, roles)
                ))
                role, ordinal, _argument = positions[index]
                cell = book.latest_ref(
                    LEXICAL_READ_BINDING,
                    (scope, int(instruction.res.id), role, ordinal),
                )
                if cell is not None and cell not in found:
                    found.append(cell)
            return tuple(found)

        # DERIVED(the ``lexical_read_binding`` cells of the first binding's
        # slots, the source's canonical cell).
        _post_derived_or_raw(
            book, REGION_CAPTURE_BINDING,
            (str(control_scope), int(region_index), value_id),
            (value_id, str(bindings[0])),
            (*read_cells(bindings[0]), source_cell),
            stage=CONTROL_SSA_REGION, mode=Mode.CONCORD,
        )
        for binding in bindings[1:]:
            binding_cells = read_cells(binding)
            # The split formal is a NOVEL mint: REGION_FORMAL_SPLIT from the
            # binding's read cells and the source's canonical cell.
            source_value = next(
                instruction.args[index]
                for instruction, index, read in reads if read == binding
            )
            formal_id = _mint_ssa_id(
                book, str(control_scope), REGION_FORMAL_SPLIT,
                (*binding_cells, source_cell),
                dtype=source_value.dtype, shape=tuple(source_value.shape or ()),
                stage=CONTROL_SSA_REGION,
            )
            for instruction, index, read in reads:
                if read != binding:
                    continue
                source = instruction.args[index]
                instruction.args[index] = SSAValue(
                    formal_id,
                    dtype=source.dtype,
                    shape=source.shape,
                    device=getattr(source, "device", None),
                )
            _post_derived_or_raw(
                book, REGION_CAPTURE_BINDING,
                (str(control_scope), int(region_index), formal_id),
                (value_id, str(binding)),
                (
                    *binding_cells, source_cell,
                    book.latest_ref(SSA_VALUE, (str(control_scope), formal_id)),
                ),
                stage=CONTROL_SSA_REGION, mode=Mode.CONCORD,
            )
            expanded.append(formal_id)
    return tuple(expanded)


def lower_control_sections_to_ssa(
    control: ControlProgram,
    *,
    hierarchy_plan: PlanClosure | None = None,
    value_concordance: IdentityPage | None = None,
    control_name: str = "planned_control",
    identity_table: Mapping[str, tuple[int, ...]] | None = None,
    function_outputs: tuple[str, ...] = (),
    function_parameters: tuple[str, ...] = (),
    lexical_read_scope: Any = None,
    shape_scope: Any = None,
    value_dtypes: Mapping[int, str] | None = None,
    value_shapes: Mapping[int, tuple[int, ...]] | None = None,
    constant_values: Mapping[int, Any] | None = None,
    required_output_value_ids: tuple[int, ...] = (),
    region_output_value_ids: Mapping[int, tuple[int, ...]] | None = None,
    record_field_write_value_ids: tuple[int, ...] = (),
    self_value_id: int | None = None,
    field_ops: tuple[tuple[str, int, int], ...] = (),
    field_write_state_cells: tuple[Any, ...] = (),
    field_const_sources: Mapping[int, Any] | None = None,
    field_count: int = 0,
    field_names: tuple[str, ...] = (),
    record_identity: str | None = None,
    record_field_dtypes: Mapping[str, str] | None = None,
    record_field_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    record_field_mutability: Mapping[str, bool] | None = None,
    sequence_initializations: tuple[tuple[int, str, int], ...] = (),
    field_aliases: tuple[tuple[int, int], ...] = (),
    sequence_declarations: tuple[tuple[int, str, int, bool], ...] = (),
    sequence_capacity_sources: tuple[tuple[int, int], ...] = (),
    sequence_column_dtypes: Mapping[int, tuple[str, ...]] | None = None,
    sequence_record_identities: Mapping[int, str] | None = None,
    sequence_row_record_slots: Mapping[
        int, tuple[tuple[int, str, int], ...]
    ] | None = None,
    callsite_projection_ids: Mapping[int, tuple[int, ...]] | None = None,
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
    nested_tensor_sequence_ids: tuple[int, ...] = (),
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
    progress: Callable[[str], None] | None = None,
    parameter_value_abi: Mapping[str, Any] | None = None,
    parameter_record_abi: Mapping[str, Any] | None = None,
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

    def report(message: str) -> None:
        if progress is not None:
            progress(f"ssa-section {control_name}: {message}")

    report(
        f"start; regions={len(tuple(control.region_indices))} "
        f"hierarchy_items={len(tuple(getattr(hierarchy_plan, 'items', ()) or ()))}"
    )

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
    # One expansion per planned region, shared by every stage below, so all
    # stages agree on the ids a region defines (including the temporaries
    # its lowering coins, which are allocated plan-wide and never collide
    # with an authored id or with another region's temporaries).
    # Every value id this function already has in play, whether or not the
    # hierarchy plan mentions it (control-only graph values never do).
    known_value_ids: set[int] = {0}
    for history in (identity_table or {}).values():
        known_value_ids.update(int(value_id) for value_id in history)
    known_value_ids.update(map(int, control_dependency_value_ids(control)))
    known_value_ids.update(int(value_id) for _kind, value_id, _slot in field_ops)
    if self_value_id is not None:
        known_value_ids.add(int(self_value_id))
    known_value_ids.update(map(int, required_output_value_ids))
    known_value_ids.update(map(int, record_field_write_value_ids))
    for region_outputs in (region_output_value_ids or {}).values():
        known_value_ids.update(map(int, region_outputs))
    planning_aliases = (
        value_concordance.alias_bindings(control_name)
        if value_concordance is not None else {}
    )
    for mapping in (value_dtypes, value_shapes, constant_values,
                    planning_aliases, sequence_length_values,
                    joined_singleton_values, field_const_sources):
        known_value_ids.update(int(value_id) for value_id in (mapping or {}))
    known_value_ids.update(
        int(value_id) for value_id in planning_aliases.values()
    )
    known_value_ids.update(
        int(sequence_id)
        for sequence_id, _policy, _columns, _writable in sequence_declarations
    )
    known_value_ids.update(
        int(sequence_id) for sequence_id, _policy, _columns in sequence_initializations
    )
    for group in (table_lookups, table_stores, table_deletions, field_aliases,
                  sequence_memberships, nested_record_fields, sequence_augassigns):
        for entry in group:
            for leaf in entry:
                if isinstance(leaf, bool) or not isinstance(leaf, int):
                    continue
                known_value_ids.add(int(leaf))
    known_value_ids.update(map(int, (*retained_sequence_ids, *nested_sequence_ids,
                                     *joined_sequence_ids, *source_sequence_ids)))
    if os.environ.get("TURING_DEBUG_REGION_OUTPUTS"):
        print(f"DEBUG-REGION-WATERMARK {control_name}: known_max={max(known_value_ids)}",
              file=sys.stderr, flush=True)
    expanded_plan_regions = (
        expand_plan_regions(
            hierarchy_plan,
            first_free_value_id=max(known_value_ids) + 1,
            function_scope=control_name,
            lexical_read_scope=lexical_read_scope,
        )
        if hierarchy_plan is not None else {}
    )

    def region_instructions(closure: "PlanClosure") -> list[Instr]:
        key = (str(closure.name), int(closure.closure_id))
        if key not in expanded_plan_regions:
            raise RuntimeError(
                f"planned region {closure.name} (closure {closure.closure_id}) "
                "is not a top-level region of the hierarchy plan; its "
                "instructions have no plan-wide expansion to share"
            )
        return copy_region_instructions(expanded_plan_regions[key])

    def lexical_mapping_writes(block):
        mutations = list(getattr(block, "sequence_mutations", ()))
        if isinstance(block, SequenceMutationBlock):
            mutations.append(block.mutation)
        for mutation in mutations:
            if mutation.operator == "update" and mutation.argument_kind == "mapping_items":
                yield int(mutation.effect_node_id)
        for child in getattr(block, "blocks", ()):
            yield from lexical_mapping_writes(child)
        for name in ("condition", "body", "orelse", "callee"):
            child = getattr(block, name, None)
            if child is not None:
                yield from lexical_mapping_writes(child)

    lexical_mapping_write_ids = set(lexical_mapping_writes(control.root))
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
            planned_instructions = region_instructions(planned)
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
        # method input.  Its uses name the candidates; the declared fact is
        # STORAGE IDENTITY: a candidate is a payload only when its
        # ``storage_root`` row reaches a formal's storage cell along DERIVED
        # edges (a view, index or alias of input storage).  ``m = x * 1.0``
        # is new storage (a PRODUCED root) and mints no row column; this
        # used to be decided by "no known shape yet", which classified it a
        # payload and left an unbound row formal (Woodshop ``_resolve_floor``:
        # ``velocity = momentum / mass; if velocity[2] < 0.0: momentum[2]``).
        from .identity_concordance import current_identity_book

        storage_book = current_identity_book()
        storage_scope = storage_book.mint_scope(
            f"storage_root:{control_name}", STORAGE_ROOT_STAGE,
        )
        storage_formals: dict[int, Any] = {}
        formal_names: dict[int, str] = {}
        for parameter_name in function_parameters:
            history = (identity_table or {}).get(str(parameter_name), ())
            if not history:
                continue
            formal_names[int(history[0])] = str(parameter_name)
            storage_formals[int(history[0])] = (
                None if lexical_read_scope is None
                else storage_book.latest_ref(
                    NAME_BINDING,
                    (lexical_read_scope, str(parameter_name), 0),
                )
            )
        if self_value_id is not None:
            storage_formals.setdefault(int(self_value_id), None)
            formal_names.setdefault(int(self_value_id), "self")
        # A formal-rooted candidate needs a row column only when its formal
        # is declared as a record / row handle carrying the payload (or
        # nothing declares it).  A declared span is its own binding: it is
        # already a named formal.  The ``parameter_abi_kind`` row is the
        # authority (``y = x * a; y[0] = x[1] + b`` with ``x`` a span).
        abi_kinds = post_parameter_abi_kinds(
            storage_book, lexical_read_scope, formal_names.values(),
            value_abi=dict(parameter_value_abi or {}),
            record_abi=dict(parameter_record_abi or {}),
        )
        abi_kind_page = storage_book.page(PARAMETER_ABI_KIND)

        def carries_payload(root_cell: Ref | None) -> bool:
            if root_cell is None:
                return False
            name = formal_names.get(int(root_cell.row[1]))
            cell = None if name is None else abi_kinds.get(name)
            kind = None if cell is None else abi_kind_page.latest(cell.row)
            return not isinstance(kind, ParameterAbiKind) or (
                kind is ParameterAbiKind.RECORD
            )

        payload_candidates = (
            (indexed_base_ids & scalar_use_ids) - declared_sequence_ids
        )
        storage_roots = _post_storage_roots(
            storage_book, storage_scope, lexical_read_scope,
            formals=storage_formals, instructions=semantic_instructions,
            aliases=planning_aliases, wanted=sorted(payload_candidates),
        )
        variant_projected_target_ids.update(
            value_id for value_id in payload_candidates
            if carries_payload(
                _storage_formal_root(storage_book, storage_roots[value_id])
            )
        )
        for planned in hierarchy_plan.items:
            if not (
                isinstance(planned, PlanClosure)
                and planned.name.startswith("region_")
            ):
                continue
            region_index = int(planned.name.rsplit("_", 1)[1])
            instructions = region_instructions(planned)
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
        # Mod and FloorDiv are integer only when integer evidence reaches
        # them (an index result, or all-integer operands), like Add/Mul:
        # Python ``%`` and ``//`` on floats are float.  As unconditional
        # integer seeds they retyped the whole float cone of a float Mod to
        # int64: orbital_game_phasing ``t36 = t32 % (2*pi)`` made the
        # sign() Select feeding t32 an int64 region result, its 1.0 arm was
        # cast to i64 and read back by where_double as 4.9e-324, and
        # t_wait came out 0 instead of 4995.3 s (2026-10-03,
        # docs/concordance_census/CONTINUATION_phasing_regression.md).
        integer_ops = {
            "And", "Or", "Xor", "Shl", "Shr",
            "bitand", "bitor", "bitxor", "shl", "shr",
        }
        arithmetic_ops = {
            "Add", "Sub", "Mul", "add", "sub", "mul",
            "FloorDiv", "Mod", "floordiv", "mod",
        }
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
    sequence_query_arena_ids: set[int] = set()
    sequence_queries: list[SequenceQueryBlock] = []

    def collect_sequence_query_results(block: ControlBlock) -> None:
        if isinstance(block, SequenceQueryBlock):
            sequence_queries.append(block)
            sequence_query_result_ids.add(int(block.result_value_id))
            sequence_query_result_ids.update(map(int, block.result_alias_ids))
            sequence_query_arena_ids.add(int(block.sequence_value_id))
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
    if value_concordance is None:
        value_concordance = IdentityPage("planning_value_concordance")

    # ``ControlProgram.value_aliases`` is an equivalence ledger assembled by
    # lexical control composition.  Once a component contains exactly one
    # declared resident sequence, its other spellings are proven logical
    # versions of that sequence.  This is NOT necessarily physical in-place
    # storage: a conditional ``where`` can compute the next whole mapping.
    # Publish it on its own concordance page so region output construction does
    # not mistake a computed replacement for an IndexedStore, then give it to
    # control SSA only after region signatures are settled.
    declared_sequence_arenas = {
        *(int(sequence_id) for sequence_id, *_rest in sequence_declarations),
        *(int(sequence_id) for sequence_id, *_rest in sequence_initializations),
        *map(int, source_sequence_ids),
        *map(int, retained_sequence_ids),
        *map(int, nested_sequence_ids),
        *map(int, joined_sequence_ids),
    }
    from .identity_concordance import current_identity_book

    control_value_concordance = current_identity_book().page(
        "control_value_concordance"
    )
    # A source method can be lowered more than once while specializations are
    # assembled.  Its authored value ids repeat, while the physical metadata
    # ids minted by each lowering do not.  Scope tensor rows to this exact
    # control instance so one lowering cannot read another's shape address.
    # The scope is numbered by the book (``scope_registry``), so each
    # lowering of one method owns distinct rows and the rows carry no
    # process object id (census 75, section 8).
    _control_label, _control_serial = current_identity_book().mint_scope(
        f"control:{control_name}"
    )
    tensor_shape_concordance_scope = (
        f"{control_name}@control:{_control_serial}"
    )
    # Row ``(control scope, value id)`` -> the dtype the control program
    # declares for a uniform it reads (loop-bound leaves are ints).  One
    # fact, read by region formal typing and by the control's own argument.
    from .identity_concordance import current_identity_book as _book

    uniform_dtype_page = _book().page("control_uniform_dtype")
    # Row ``(control scope, value id)`` -> the scalar dtype the producing
    # region computes for that value.
    region_dtype_page = _book().page("region_value_dtype")

    def _graph_cell(value_id: Any) -> Any:
        """The ``canonical_value`` cell of a graph id in this function."""

        if lexical_read_scope is None or value_id is None:
            return None
        row = (tuple(lexical_read_scope), int(value_id))
        # A node a specialized copy added after the canonical relabel has its
        # ``ingestion_value`` row in the copy's scope and no canonical row.
        return _book().latest_ref(CANONICAL_VALUE, row) or _book().latest_ref(
            INGESTION_VALUE, row,
        )

    for uniform in getattr(control, "uniforms", ()) or ():
        # DERIVED(the uniform's declaration cell: its ``scalar_parameter``
        # row when declared, else its canonical cell).
        _post_derived_or_raw(
            _book(), CONTROL_UNIFORM_DTYPE,
            (str(tensor_shape_concordance_scope), int(uniform.value_id)),
            str(uniform.dtype),
            (
                _book().latest_ref(
                    SCALAR_PARAMETER,
                    (str(control_name), int(uniform.value_id)),
                ),
                _graph_cell(uniform.value_id),
            ),
            stage=CONTROL_SSA_ENTRY, mode=Mode.CONCORD,
        )
    # ``np.asarray`` reaches the plan as canonical ``tensor``.  When its
    # requested dtype already equals the operand dtype it is a view of the
    # same storage identity, not a new arena.  Publish that fact before
    # region captures are built so a later region reads the resident
    # sequence root directly instead of receiving a row-shaped stand-in.
    # This is the same value concordance used for indexed-store versions and
    # loop-carried storage; no shape or name matching participates.
    for expanded in expanded_plan_regions.values():
        for instruction in expanded:
            if (
                instruction.res is None
                or len(instruction.args) != 1
                or str(instruction.op).casefold() != "cast"
                or str((instruction.attributes or {}).get(
                    "source_operator", ""
                )).casefold() != "tensor"
            ):
                continue
            source = instruction.args[0]
            result = instruction.res
            source_dtype = str(source.dtype or "unknown").casefold()
            result_dtype = str(result.dtype or "unknown").casefold()
            if (
                source_dtype in {"", "none", "unknown"}
                or result_dtype != source_dtype
            ):
                continue
            alias_id, resident_id = int(result.id), int(source.id)
            incumbent = value_concordance.resolve_alias(
                control_name, alias_id
            )
            resident_root = value_concordance.resolve_alias(
                control_name, resident_id
            )
            if incumbent == resident_root:
                continue
            if incumbent != alias_id:
                raise ValueError(
                    "tensor normalization identity disagrees with planning "
                    f"concordance for {control_name!r}: result={alias_id} "
                    f"already resolves to {incumbent}, operand resolves to "
                    f"{resident_root}"
                )
            value_concordance.bind_alias(
                control_name, alias_id, resident_root
            )
    alias_neighbors: dict[int, set[int]] = {}
    for left, right in control.value_aliases:
        left, right = int(left), int(right)
        alias_neighbors.setdefault(left, set()).add(right)
        alias_neighbors.setdefault(right, set()).add(left)
    visited_aliases: set[int] = set()
    for seed in tuple(alias_neighbors):
        if seed in visited_aliases:
            continue
        component, frontier = set(), [seed]
        while frontier:
            value_id = frontier.pop()
            if value_id in component:
                continue
            component.add(value_id)
            frontier.extend(alias_neighbors.get(value_id, ()))
        visited_aliases.update(component)
        residents = component & declared_sequence_arenas
        if len(residents) != 1:
            continue
        resident_id = next(iter(residents))
        for alias_id in sorted(component - {resident_id}):
            incumbent = control_value_concordance.latest(
                (control_name, int(alias_id))
            )
            if incumbent is None:
                # DERIVED(alias node's and resident's canonical cells); the
                # planning alias row that joined them is the planner's page.
                _post_derived_or_raw(
                    _book(), CONTROL_VALUE_CONCORDANCE,
                    (control_name, int(alias_id)), int(resident_id),
                    (_graph_cell(alias_id), _graph_cell(resident_id)),
                    stage=CONTROL_SSA_ENTRY, mode=Mode.REVISE,
                )
            elif control_value_concordance.resolve_alias(
                control_name, int(alias_id)
            ) != control_value_concordance.resolve_alias(
                control_name, int(resident_id)
            ):
                raise ValueError(
                    "control sequence identity disagrees with planning "
                    f"concordance for {control_name!r}: "
                    f"alias={alias_id}, control_resident={resident_id}, "
                    f"concordance_resident={incumbent}"
                )
    from .identity_concordance import resolved_concordant_alias_bindings

    region_value_aliases = resolved_concordant_alias_bindings(
        control_name,
        control_value_concordance.alias_bindings(control_name),
        page=value_concordance,
    )
    # Publish the resolved roots back to the live authority. Later passes and
    # the durable function snapshot must see the same complete identity, not
    # whichever half of the chain their stage happened to author.
    for alias_id, resident_id in region_value_aliases.items():
        if value_concordance.latest(
            (control_name, int(alias_id))
        ) != int(resident_id):
            value_concordance.bind_alias(
                control_name, int(alias_id), int(resident_id)
            )

    def bind_resident(alias_id: int, resident_id: int) -> None:
        """Publish a planning alias, then mirror the concorded fact for SSA."""
        alias_id, resident_id = int(alias_id), int(resident_id)
        alias_root = value_concordance.resolve_alias(control_name, alias_id)
        resident_root = value_concordance.resolve_alias(
            control_name, resident_id
        )
        if alias_root == resident_root:
            # A later stage may encounter the same resident relationship in
            # the opposite syntactic direction (loop-result versus final
            # IndexedStore).  It is already one concordance row; recording the
            # reverse edge would manufacture a cycle from an agreement.
            region_value_aliases[alias_id] = resident_root
            return
        if alias_root != alias_id:
            raise ValueError(
                "planning identity concordance disagreement for "
                f"{control_name!r}: alias {alias_id} already resolves to "
                f"{alias_root}, cannot bind it to {resident_root}"
            )
        value_concordance.bind_alias(
            control_name, alias_id, resident_root
        )
        region_value_aliases[alias_id] = resident_root

    def resident_root(value_id: int) -> int:
        """Canonical planning identity for one value occurrence."""

        return value_concordance.resolve_alias(control_name, value_id)
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
            for instruction in region_instructions(item)
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
            bind_resident(result_id, sequence_id)
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
        instructions: Any,
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
            planned_instructions = region_instructions(planned)
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
            instructions = region_instructions(region)
            report(
                f"region {region_index}: lowering {len(instructions)} "
                "planned instruction(s)"
            )
            # A resident generator query (currently ``sum(1 for ...)`` or
            # ``next(..., default)``) is emitted in lexical control beside its
            # producer loop. Remove only that replaced operator from the
            # numerical region; downstream arithmetic/compare instructions
            # remain and consume the query's exact result identity.
            instructions = [
                instruction for instruction in instructions
                if instruction.res is None
                or (int(instruction.res.id) not in sequence_query_result_ids
                    and not (
                        instruction.op == "Const"
                        and int(instruction.res.id) in sequence_query_arena_ids
                        and isinstance(instruction.attributes.get("value", instruction.attributes.get("constant")), list)
                        and not instruction.attributes.get("value", instruction.attributes.get("constant"))
                    ))
            ]
            replaced_maximum_calls = {
                int(query.source_call_node_id)
                for query in sequence_queries
                if query.operation == "maximum"
                and query.source_call_node_id is not None
                and int(query.source_call_node_id) in set(map(
                    int,
                    (region_output_value_ids or {}).get(region_index, ()),
                ))
            }
            if replaced_maximum_calls:
                # Variadic Python max is flattened into a chain of Max
                # instructions in one numerical region.  The resident query
                # replaces that whole chain, while its scalar prefix operands
                # (often Div results) remain real region outputs consumed by
                # the query after the generator completes.
                instructions = [
                    instruction for instruction in instructions
                    if not (
                        instruction.op == "Max"
                        and instruction.attributes.get("extraction_identity")
                        == "builtins.max"
                    )
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
                    bind_resident(result_id, sequence_id)
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
                    bind_resident(result_id, destination_id)
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
                    bind_resident(result_id, result_id)
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
                    bind_resident(result_id, destination_id)
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
                    bind_resident(result_id, destination_id)
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
                    bind_resident(result_id, sequence_id)
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
                    bind_resident(result_id, destination_id)
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
                        bind_resident(next(
                            result_id for result_id, contract
                            in inplace_pack_by_result.items()
                            if contract == prior_pack
                            and result_id in {
                                int(value.id) for value in instruction.args
                            }
                        ), sequence_id)
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
                bind_resident(result_id, sequence_id)
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
                    bind_resident(
                        int(instruction.res.id), int(instruction.args[0].id),
                    )
            instructions = [instruction for instruction in instructions
                            if not (instruction.op == "IndexedStore"
                                    and instruction.res is not None
                                    and int(instruction.res.id) in lexical_mapping_write_ids)]
            if lexical_mapping_write_ids:
                planned_region_instructions[region_index] = tuple(instructions)
            if not instructions:
                handled_table_region_indices.add(region_index)
                continue
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
            # Physical parameter/field ABI applies inside numerical callees
            # too. Applying this only when constructing the coordinator let
            # its byte-valued bool field be read as a double by a region.
            # A control uniform's dtype is a contract of the control program
            # (a ``range`` bound leaf is an int); the book row committed at
            # entry is read here so the region formal and the control caller
            # agree.  A declared physical ABI dtype below still wins.
            for value_id in instruction_values:
                uniform_dtype = uniform_dtype_page.latest(
                    (str(tensor_shape_concordance_scope), int(value_id))
                )
                if uniform_dtype is not None:
                    previous = region_value_meta.get(int(value_id))
                    region_value_meta[int(value_id)] = Meta(
                        () if previous is None else tuple(previous.shape),
                        str(uniform_dtype),
                    )
            # A value an earlier region produced has the dtype that region
            # typed it with (``region_value_dtype``); the planner's generic
            # capture guess must not retype it at this formal.
            for value_id in instruction_values:
                produced_dtype = region_dtype_page.latest(
                    (str(tensor_shape_concordance_scope), int(value_id))
                )
                if produced_dtype is not None:
                    previous = region_value_meta.get(int(value_id))
                    region_value_meta[int(value_id)] = Meta(
                        () if previous is None else tuple(previous.shape),
                        str(produced_dtype),
                    )
            for value_id, dtype in (value_dtypes or {}).items():
                if int(value_id) in instruction_values:
                    previous = region_value_meta.get(int(value_id))
                    region_value_meta[int(value_id)] = Meta(
                        tuple((value_shapes or {}).get(int(value_id),
                              () if previous is None else previous.shape)), str(dtype),
                    )
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
            # Captures cross from the authored control frame into a planned
            # region. Resolve them through the planning concordance before
            # either side's ABI is constructed. Otherwise an exact-return
            # helper can leave the region formal under its result spelling
            # while the caller has already resolved the same occurrence to
            # the returned argument spelling.
            capture_rebindings = {
                int(value_id): resident_root(int(value_id))
                for value_id in effective_captures
                if resident_root(int(value_id)) != int(value_id)
            }
            if capture_rebindings:
                for instruction in instructions:
                    instruction.args = [
                        SSAValue(
                            capture_rebindings[int(argument.id)],
                            dtype=argument.dtype,
                            shape=tuple(argument.shape or ()),
                            device=argument.device,
                            accounting=dict(argument.accounting or {}),
                        )
                        if int(argument.id) in capture_rebindings
                        else argument
                        for argument in instruction.args
                    ]
                effective_captures = tuple(dict.fromkeys(
                    resident_root(int(value_id))
                    for value_id in effective_captures
                ))

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
            # A canonical literal (one graph node, a known value) is owned
            # by the control function, which materializes it itself
            # (``_materialize_control_constants``).  The planner
            # rematerializes such a literal privately inside every region
            # that uses it; those copies are never that region's outputs.
            # Publishing them made four regions each define the one id
            # (``step_with_dt_control_used`` 342 x4).
            control_owned_literals = {
                int(instr.res.id)
                for instr in instructions
                if instr.res is not None
                and str(instr.op).casefold() in {"const", "constant"}
                and int(instr.res.id) in (constant_values or {})
            }
            outputs = tuple(sorted(
                value_id for value_id in produced
                if value_id not in control_owned_literals
                and (
                    (
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
                    and (
                        tuple(argument.shape or ())
                        or (argument.accounting or {}).get(
                            "program_abi_storage"
                        ) == "span"
                        or (argument.accounting or {}).get(
                            "tensor_metadata_state"
                        ) == "dynamic"
                    )
                )
                contracts = {
                    (
                        tuple(value.shape),
                        str(value.dtype or ""),
                        (value.accounting or {}).get("program_abi_storage"),
                        (value.accounting or {}).get("program_abi_rank"),
                        (value.accounting or {}).get("tensor_metadata_state"),
                        (value.accounting or {}).get("tensor_shape_value_id"),
                        (value.accounting or {}).get("tensor_rank_value_id"),
                        (value.accounting or {}).get(
                            "tensor_element_count_value_id"
                        ),
                    )
                    for value in occurrences
                }
                # A sole concrete use is the exact ordered region view. When
                # several reshapes share the storage id, keep the canonical
                # formal neutral; the individual op occurrences above retain
                # their own views.
                if len(contracts) == 1:
                    return occurrences[0]
                return canonical

            # Python integer arithmetic stays integer: a scalar result whose
            # operands are all integers and whose dtype is only the planner's
            # generic guess is an integer, decided here, once, before the
            # control caller and later regions read it (``n + k`` indexing
            # an array; its caller once loaded the int result as a double).
            integer_dtypes = {"int", "int64", "int32", "int16", "int8"}
            integer_ops = {
                "add", "sub", "mul", "floordiv", "mod", "neg",
                "min", "max", "abs",
            }
            declared_ids = {int(value_id) for value_id in (value_dtypes or {})}
            integer_results: dict[int, str] = {}
            for instruction in instructions:
                result = instruction.res
                if (
                    result is None
                    or str(instruction.op).casefold() not in integer_ops
                    or not instruction.args
                    or tuple(result.shape or ())
                    or int(result.id) in declared_ids
                    or str(result.dtype or "") in integer_dtypes
                ):
                    continue
                operand_dtypes = [
                    integer_results.get(int(argument.id), str(argument.dtype or ""))
                    for argument in instruction.args
                ]
                if all(
                    dtype in integer_dtypes and not tuple(argument.shape or ())
                    for dtype, argument in zip(operand_dtypes, instruction.args)
                ):
                    integer_results[int(result.id)] = "int64"
            if integer_results:
                for instruction in instructions:
                    values = [*instruction.args]
                    if instruction.res is not None:
                        values.append(instruction.res)
                    for value in values:
                        dtype = integer_results.get(int(value.id))
                        if dtype is not None:
                            value.dtype = dtype
                for value_id, dtype in integer_results.items():
                    if value_id in region_values:
                        region_values[value_id].dtype = dtype
                    previous = region_value_meta.get(value_id)
                    region_value_meta[value_id] = Meta(
                        () if previous is None else tuple(previous.shape),
                        dtype,
                    )
            # Each scalar result's dtype as this region computes it is the
            # fact later regions read for the same value.
            for instruction in instructions:
                result = instruction.res
                if (
                    result is None
                    or result.dtype is None
                    or tuple(result.shape or ())
                    or str(result.dtype) in {"ptr", "unknown"}
                ):
                    continue
                row = (str(tensor_shape_concordance_scope), int(result.id))
                if region_dtype_page.latest(row) != str(result.dtype):
                    # REVISE DERIVED(the value's canonical cell, the
                    # incumbent dtype cell): a later region's different
                    # dtype is a retype the audit sees.
                    _post_derived_or_raw(
                        _book(), REGION_VALUE_DTYPE, row, str(result.dtype),
                        (
                            _book().latest_ref(REGION_VALUE_DTYPE, row),
                            _graph_cell(result.id),
                        ),
                        stage=CONTROL_SSA_REGION, mode=Mode.REVISE,
                    )
            effective_captures = _split_region_captures_by_binding(
                lexical_read_scope, tensor_shape_concordance_scope,
                region_index, instructions, effective_captures,
            )
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
            region_function.metadata[
                "tensor_shape_concordance_scope"
            ] = tensor_shape_concordance_scope
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
            if os.environ.get("TURING_DEBUG_REGION_OUTPUTS"):
                print(
                    f"DEBUG-REGION-OUTPUTS {control_name} {region_name}: "
                    f"produced={sorted(produced)} outputs={outputs} "
                    f"required={sorted(required_outputs & produced)} "
                    f"req-sets={ {name: sorted(set(map(int, group)) & produced) for name, group in (('plan_live', resolved_plan_live_value_ids), ('control_dep', control_dependency_value_ids(control)), ('authored_out', authored_output_value_ids), ('required_out', required_output_value_ids), ('region_out', (region_output_value_ids or {}).get(region_index, ())), ('record_write', record_field_write_value_ids))} } "
                    f"aliases={ {k: v for k, v in region_value_aliases.items() if k in produced or v in produced} } "
                    f"instrs={[(str(i.op), [int(a.id) for a in i.args], None if i.res is None else int(i.res.id)) for i in instructions]}",
                    file=sys.stderr, flush=True,
                )
            functions[region_name] = region_function
            region_callees[region_index] = region_name
            region_signatures[region_index] = (
                tuple(int(vid) for vid in effective_captures),
                outputs,
            )
            signature_cell = _post_region_signature(
                _book(), str(tensor_shape_concordance_scope), lexical_read_scope,
                region_index, region_signatures[region_index],
            )
            _post_planned_region_identities(
                _book(), str(tensor_shape_concordance_scope), region_function,
                signature_cell, _graph_cell,
            )
            _concord_region_feed_consumers(
                lexical_read_scope, tensor_shape_concordance_scope,
                region_index, instructions, effective_captures,
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
            if not any(instruction.op in {"Ret", "Br", "CondBr", "Switch", "IndirectBr"}
                       for instruction in instructions):
                # Region output maps serve legacy emitters, but repository
                # SSA consumers need an explicit publication terminator too.
                # Without it, C computes the result and returns without ever
                # writing the caller's output storage.
                definitions = {int(value.id): value for value in arguments}
                definitions.update({int(instruction.res.id): instruction.res
                                    for instruction in instructions if instruction.res is not None})
                instructions.append(Instr("Ret", [definitions[int(vid)] for vid in outputs], None))
            # Do NOT gate region ops on the repository ``Handler`` enum here: that
            # is the LLVM/repository vocabulary, but this path emits through the
            # selected target (Fortran), whose op set is broader -- e.g. it
            # renders ``equal`` as ``(a == b)`` though ``Handler`` only knows
            # ``Eq``. The target's own emit reports any op it genuinely cannot
            # express, with a message accurate to that target.
            report(
                f"region {region_index}: complete; "
                f"captures={len(effective_captures)} outputs={len(outputs)} "
                f"instructions={len(instructions)}"
            )
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
    graph_value_ids = [0, *known_value_ids]
    for history in (identity_table or {}).values():
        graph_value_ids.extend(int(value_id) for value_id in history)
    for expanded in expanded_plan_regions.values():
        for instruction in expanded:
            graph_value_ids.extend(int(value.id) for value in instruction.args)
            if instruction.res is not None:
                graph_value_ids.append(int(instruction.res.id))
    if os.environ.get("TURING_DEBUG_REGION_OUTPUTS"):
        print(f"DEBUG-REGION-WATERMARK {control_name}: graph_max={max(graph_value_ids)} "
              f"control_dep_max={max(map(int, control_dependency_value_ids(control)), default=-1)}",
              file=sys.stderr, flush=True)
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
    item_operand_page = None
    if lexical_read_scope is not None:
        from .identity_concordance import current_identity_book

        item_operand_page = current_identity_book().page("item_operand")
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
        # A captured ``item()`` IS its operand's read (``item_operand``):
        # the region consumes the operand's producer too.  Matched by id
        # alone, ``float(x.item())`` had no dependency on the region
        # producing ``x`` and was scheduled before it
        # (``_propose_dt_pen``: ``float(maximum(ratios.max(), 1.0).item())``).
        if item_operand_page is not None:
            consumed = tuple(dict.fromkeys((
                *consumed,
                *(
                    int(item[0])
                    for value_id in consumed
                    for item in (item_operand_page.latest(
                        (tuple(lexical_read_scope), int(value_id))
                    ),)
                    if item is not None
                ),
            )))
        region_dependency_signatures[int(region_index)] = (consumed, produced)

    def retained_lookup_ids(block):
        if isinstance(block, SequenceQueryBlock) and block.operation == "lookup":
            yield int(block.result_value_id)
        for child in getattr(block, "blocks", ()):
            yield from retained_lookup_ids(child)
        for name in ("condition", "body", "orelse", "callee"):
            child = getattr(block, name, None)
            if child is not None:
                yield from retained_lookup_ids(child)

    retained_table_lookup_ids = set(retained_lookup_ids(control.root))
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
            excluded_result_ids=region_scheduled_lookup_ids | retained_table_lookup_ids,
            globally_mutated_sequence_ids=(
                int(sequence_id)
                for _effect, _key, _value, sequence_id in table_stores
            ),
        )
    )
    lexical_table_lookup_ids = tuple(sorted(
        set(lexical_table_lookup_ids) | retained_table_lookup_ids
    ))
    shortfalls.extend(
        SSALoweringShortfall(
            "ssa-table", "lexical-lookup", control_name,
            f"lookup {result_id} cannot be placed: {reason}",
        )
        for result_id, reason in lookup_refusals
    )
    from .ir_identities import _PURE_REGION_OPS

    control, plan_callsite_bindings = _schedule_loop_callsites(
        control,
        hierarchy_plan,
        region_signatures,
        region_dependency_signatures,
        callsite_projection_ids=callsite_projection_ids,
        pure_region_indices=frozenset(
            index for index, instructions in planned_region_instructions.items()
            if all(instruction.op in _PURE_REGION_OPS for instruction in instructions)
        ),
    )
    # Uniform types describe a control use (for example an integer loop
    # bound), not authority to reinterpret an explicitly typed ABI buffer.
    # The same value can also feed tensor arithmetic. Preserve the declared
    # storage type at the coordinator just as region lowering does above.
    control = replace(control, uniforms=tuple(
        replace(uniform, dtype=str((value_dtypes or {}).get(
            int(uniform.value_id), uniform.dtype
        )))
        for uniform in control.uniforms
    ))
    report(
        f"control lowering start; region_functions={len(region_callees)}"
    )
    from .control_source import place_loop_carried_region_producers

    final_value_aliases = resolved_concordant_alias_bindings(
        control_name,
        region_value_aliases,
        control_value_concordance.alias_bindings(control_name),
        page=value_concordance,
    )
    tensor_shape_concordance = current_identity_book().page(
        "tensor_shape_concordance"
    )

    def final_resident(value_id: int) -> int:
        return int(final_value_aliases.get(int(value_id), int(value_id)))

    # IndexedStore is the numerical stage's exact proof that a result versions
    # resident tensor storage. Publish that storage contract for every
    # concorded spelling before lexical loop lowering temporarily rewrites
    # result-port aliases. A loop backedge can then recognize the resident as
    # a span even when no stage-local Meta uses the final loop-result id.
    for instructions in planned_region_instructions.values():
        for instruction in instructions:
            if (
                str(instruction.op).casefold() != "indexedstore"
                or not instruction.args
                or instruction.res is None
            ):
                continue
            base, result = instruction.args[0], instruction.res
            resident_id = final_resident(base.id)
            rank = max(
                len(tuple(base.shape or ())),
                len(tuple(result.shape or ())),
                int((base.accounting or {}).get("program_abi_rank", 0) or 0),
                int((result.accounting or {}).get(
                    "program_abi_rank", 0
                ) or 0),
            )
            contract = {
                "program_abi_storage": "span",
                "program_abi_rank": int(rank),
                "tensor_metadata_state": (
                    "static" if rank > 0 else "unknown"
                ),
                "shape": tuple(base.shape or result.shape or ()),
                "source": "planned-region-indexed-store",
            }
            spellings = (
                int(base.id), int(result.id), int(resident_id),
                *(
                    int(alias_id)
                    for alias_id, target_id in final_value_aliases.items()
                    if int(target_id) == resident_id
                ),
            )
            for value_id in dict.fromkeys(spellings):
                incumbent = tensor_shape_concordance.latest((
                    tensor_shape_concordance_scope, value_id,
                ))
                committed = dict(contract)
                if incumbent is not None:
                    committed = {
                        **contract,
                        **dict(incumbent),
                        "program_abi_storage": "span",
                        "program_abi_rank": max(
                            int(rank),
                            int(dict(incumbent).get(
                                "program_abi_rank", 0
                            ) or 0),
                        ),
                    }
                # REVISE DERIVED(the incumbent contract cell when merged
                # over one, the value's canonical cell).
                _post_derived_or_raw(
                    _book(), TENSOR_SHAPE_CONCORDANCE,
                    (tensor_shape_concordance_scope, value_id), committed,
                    (
                        _book().latest_ref(
                            TENSOR_SHAPE_CONCORDANCE,
                            (tensor_shape_concordance_scope, value_id),
                        ),
                        _graph_cell(value_id),
                    ),
                    stage=CONTROL_SSA_REGION, mode=Mode.REVISE,
                )
    control = place_loop_carried_region_producers(
        control,
        {
            region_index: tuple(
                int(instruction.res.id)
                for instruction in instructions
                if instruction.res is not None
            )
            for region_index, instructions in planned_region_instructions.items()
        },
        # Region results and loop-result nodes are transient spellings of the
        # same resident tensor after an in-place store.  This map has already
        # been merged with and resolved through planning_value_concordance;
        # schedule by that identity instead of comparing stage-local ids.
        value_aliases=final_value_aliases,
        lexical_read_scope=lexical_read_scope,
    )
    # A keyed Tensor lookup is materialized by the control function while its
    # numerical uses live in planned regions.  Commit the region's logical
    # rank claim under the lookup's exact value identity, then give that fact
    # to the control builder when it binds the selected child span.  The flat
    # child arena is not rank one merely because its storage is flat.
    region_value_ranks: dict[int, int] = {}
    for region_index, instructions in planned_region_instructions.items():
        for instruction in instructions:
            for value in (
                *instruction.args,
                *((instruction.res,) if instruction.res is not None else ()),
            ):
                value_id = int(value.id)
                if value_id not in table_lookup_result_ids:
                    continue
                rank = max(
                    len(tuple(value.shape or ())),
                    int((value.accounting or {}).get(
                        "program_abi_rank", 0
                    ) or 0),
                )
                if rank <= 0:
                    continue
                _post_derived_or_raw(
                    _book(), TENSOR_SHAPE_CONCORDANCE,
                    (tensor_shape_concordance_scope, value_id),
                    {
                        "program_abi_storage": "span",
                        "program_abi_rank": int(rank),
                        "tensor_metadata_state": "dynamic",
                        "tensor_shape_value_id": None,
                        "tensor_rank_value_id": None,
                        "tensor_element_count_value_id": None,
                        "source": "planned-region-rank",
                    },
                    (
                        _book().latest_ref(
                            TENSOR_SHAPE_CONCORDANCE,
                            (tensor_shape_concordance_scope, value_id),
                        ),
                        _graph_cell(value_id),
                    ),
                    stage=CONTROL_SSA_REGION, mode=Mode.REVISE,
                )
                region_value_ranks[value_id] = max(
                    int(rank), region_value_ranks.get(value_id, 0)
                )
    control_function, control_shortfalls = lower_control_program_to_ssa(
        control,
        function_name=control_name,
        first_value_id=max(graph_value_ids) + 1,
        region_callees=region_callees,
        region_signatures=region_signatures,
        region_feed_meta=region_feed_meta,
        region_value_meta=region_value_meta,
        region_value_ranks=region_value_ranks,
        tensor_shape_concordance_scope=tensor_shape_concordance_scope,
        lexical_read_scope=lexical_read_scope,
        shape_scope=shape_scope,
        plan_callsite_bindings=plan_callsite_bindings,
        value_aliases=final_value_aliases,
        inout_value_ids=tuple(map(int, record_field_write_value_ids)),
        constant_value_ids=tuple(map(int, constant_values or {})),
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
        sequence_row_record_slots=sequence_row_record_slots,
        source_sequence_ids=source_sequence_ids,
        sequence_memberships=sequence_memberships,
        table_lookups=table_lookups,
        lexical_table_lookup_result_ids=lexical_table_lookup_ids,
        table_lookup_defaults=table_lookup_defaults,
        table_stores=table_stores,
        table_deletions=table_deletions,
        retained_sequence_ids=retained_sequence_ids,
        nested_sequence_ids=nested_sequence_ids,
        nested_tensor_sequence_ids=nested_tensor_sequence_ids,
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
    report(
        f"control lowering complete; blocks={len(control_function.blocks)} "
        f"shortfalls={len(control_shortfalls)}"
    )
    control_function.metadata[
        "tensor_shape_concordance_scope"
    ] = tensor_shape_concordance_scope
    # The ProcessGraph owns this exact collection->iterable edge.  Publish it
    # before repository tensor lowering below; attaching it only after this
    # function returns makes dynamic collection extents invisible at the one
    # phase that must turn ``rows[:, j]`` into bounded index arithmetic.
    control_function.metadata["sequence_capacity_sources"] = tuple(
        (int(sequence_id), int(source_id))
        for sequence_id, source_id in sequence_capacity_sources
    )
    debug_region_output_loads(control_function, "after-control-lowering")
    control_function = _materialize_control_constants(
        control_function,
        constant_values or {},
        value_dtypes=value_dtypes or {},
    )
    debug_region_output_loads(control_function, "after-constants")
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
    declared_contracts = {
        str(name): dict(contract)
        for name, contract in dict(record_field_contracts or {}).items()
    }
    # Reference is a physical ABI storage class, not an untyped scalar.  A
    # static-reference write also proves that class for source-derived records;
    # an explicit ProgramABI declaration proves it even when this particular
    # method only reads the field or writes a value produced elsewhere.
    reference_slots = {
        int(slot) for slot, name in enumerate(field_names)
        if str((declared_contracts.get(name) or {}).get("storage") or "")
        == "reference"
    } | {
        int(slot)
        for kind, value_id, slot in field_ops
        if kind == "write"
        and isinstance((field_const_sources or {}).get(int(value_id)), Mapping)
        and (field_const_sources or {})[int(value_id)].get(
            "ssa_reference_identity"
        ) is not None
    }
    span_slots = {
        int(slot) for slot, name in enumerate(field_names)
        if str((declared_contracts.get(name) or {}).get("storage") or "")
        == "span"
    }
    # A nested record field is a descriptor correlation, never a slot: its
    # leaves are the physical storage (see ``nested_record_by_slot`` below).
    from .identity_concordance import current_identity_book

    slot_storage_page = current_identity_book().page(
        "field_slot_storage_concordance"
    )
    nested_record_slots = {
        int(slot)
        for slot, identity, _value_id in nested_record_fields
        if slot_storage_page.concord(
            (
                str(control_function.name),
                str(field_names[int(slot)])
                if 0 <= int(slot) < len(field_names) else int(slot),
            ),
            ("nested_record", str(identity)),
        ) == ("nested_record", str(identity))
    }
    scalar_slots = tuple(sorted({
        int(slot)
        for _kind, _value_id, slot in field_ops
        if int(slot) not in sequence_field_slots
        and int(slot) not in span_slots
        and int(slot) not in nested_record_slots
    }))
    compact_slot = {slot: index for index, slot in enumerate(scalar_slots)}
    scalar_field_ops = tuple(
        (kind, value_id, compact_slot[slot])
        for kind, value_id, slot in field_ops
        if slot in compact_slot
    )
    # The reducer WRITTEN cell of each scalar ``"write"`` op, aligned with the
    # write ops of ``scalar_field_ops`` (empty when the shell supplied none).
    _write_ops = [op for op in field_ops if op[0] == "write"]
    scalar_write_state_cells = (
        tuple(
            cell for op, cell in zip(_write_ops, field_write_state_cells)
            if op[2] in compact_slot
        )
        if len(field_write_state_cells) == len(_write_ops) else ()
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
        if slot in compact_slot
        and slot not in reference_slots
        and name not in declared_field_dtypes
    )
    if scalar_slots and declared_field_dtypes and undeclared_scalar_fields:
        raise ValueError(
            f"record scalar slot ABI lacks dtype in {control_function.name} "
            f"for record {record_identity!r}: "
            + ", ".join(undeclared_scalar_fields)
            + f"; declared={tuple(sorted(declared_field_dtypes))!r}"
        )
    receiver_scalar_dtype = (
        sorted(scalar_dtypes)[0] if scalar_dtypes else "float64"
    )
    scalar_slot_dtypes = {
        compact_slot[old_slot]: (
            "opaque_ref" if old_slot in reference_slots
            else declared_field_dtypes.get(name, receiver_scalar_dtype)
        )
        for old_slot, name in enumerate(field_names)
        if old_slot in compact_slot
    }
    scalar_field_locations: dict[int, tuple[int, int, str]] = {}
    scalar_field_accounting = {
        compact_slot[old_slot]: {
            "program_abi_record": str(record_identity),
            "program_abi_parameter": "self",
            "program_abi_field": str(field_names[old_slot]),
            "program_abi_storage": (
                "reference" if old_slot in reference_slots else "scalar"
            ),
            "program_abi_rank": 0,
            "program_abi_mutable": bool(
                (record_field_mutability or {}).get(field_names[old_slot], False)
            ),
            "program_abi_field_written": any(
                kind == "write" and int(slot) == int(old_slot)
                for kind, _value_id, slot in field_ops
            ),
        }
        for old_slot in scalar_slots
    }
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
            field_write_state_cells=scalar_write_state_cells,
            lexical_read_scope=lexical_read_scope,
            field_const_sources=field_const_sources or {},
            field_count=len(scalar_slots),
            output_value_ids=output_value_ids,
            dtype=receiver_scalar_dtype,
            field_dtypes=scalar_slot_dtypes,
            separate_field_storage=bool(declared_contracts),
            field_accounting=scalar_field_accounting,
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
        span_field_values = {
            int(slot): int(value_id)
            for kind, value_id, slot in field_ops
            if kind == "read" and int(slot) in span_slots
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
                            *((descriptor.child_table_pool.shape_value_id,)
                              if descriptor.child_table_pool.shape_value_id is not None else ()),
                            *((descriptor.child_table_pool.rank_value_id,)
                              if descriptor.child_table_pool.rank_value_id is not None else ()),
                            *((descriptor.child_table_pool.shape_stride_value_id,)
                              if descriptor.child_table_pool.shape_stride_value_id is not None else ()),
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
            elif old_slot in span_field_values:
                contract = declared_contracts.get(name) or {}
                record_fields.append(SSARecordFieldDescriptor(
                    name,
                    SSARecordFieldStorage.SPAN,
                    storage_identity=storage_identity,
                    value_ids=(span_field_values[old_slot],),
                    dtype=contract.get("dtype"),
                    writable=bool(declared_mutability.get(name, False)),
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
                    offset=(
                        None if declared_contracts else
                        scalar_field_locations[compact_slot[old_slot]][1]
                    ),
                    dtype=(
                        "opaque_ref"
                        if old_slot in reference_slots else field_dtype
                    ),
                    writable=bool(declared_mutability.get(name, False)),
                ))
        if record_fields:
            record_tables[control_function.name] = SSARecordTable({
                int(self_value_id if self_value_id is not None else max(graph_value_ids) + 1):
                    SSARecordDescriptor(
                        int(self_value_id if self_value_id is not None else max(graph_value_ids) + 1),
                        record_identity,
                        tuple(record_fields),
                    )
            }, owner=control_function.name)
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
    # A planned region that captures a resident sequence arena may perform
    # rank-aware indexing whose leading extent is the sequence's live length.
    # Make that cell an ordinary ABI operand before tensor lowering.  The
    # sequence descriptor is the authority for the correlation; calls and
    # formals are extended together, so no stage-local ID comparison is used.
    control_sequences = sequence_tables.get(control_function.name)
    if control_sequences is not None:
        for region_function in functions.values():
            receipt = (
                region_function.metadata or {}
            ).get("source_region_integral") or {}
            if str(receipt.get("owner") or "") != control_function.name:
                continue
            additions: list[tuple[int, int]] = []
            existing_ids = {int(value.id) for value in region_function.args}
            for argument in tuple(region_function.args):
                accounting = dict(argument.accounting or {})
                storage_view = accounting.get("ssa_storage_view") or {}
                candidate_ids = {
                    int(argument.id),
                    final_resident(int(argument.id)),
                    *(
                        (int(accounting["ssa_storage_alias"]),)
                        if accounting.get("ssa_storage_alias") is not None
                        else ()
                    ),
                    *(
                        (int(storage_view["storage_value_id"]),)
                        if storage_view.get("storage_value_id") is not None
                        else ()
                    ),
                }
                descriptors = {
                    descriptor.sequence_id: descriptor
                    for candidate_id in candidate_ids
                    for descriptor in (
                        control_sequences.by_id(candidate_id),
                    )
                    if descriptor is not None
                }
                for candidate_id in candidate_ids:
                    claims = tuple(control_sequences.member_claims(candidate_id))
                    sequence_ids = {
                        int(sequence_id)
                        for sequence_id, role in claims
                        if tuple(role) == ("column", 0)
                    }
                    descriptors.update({
                        sequence_id: control_sequences.by_id(sequence_id)
                        for sequence_id in sequence_ids
                        if control_sequences.by_id(sequence_id) is not None
                    })
                if len(descriptors) != 1:
                    continue
                descriptor = next(iter(descriptors.values()))
                if len(descriptor.column_value_ids) == 1:
                    # The length cell and the row layout are two parts of one
                    # sequence descriptor.  Planned regions previously
                    # received the former while their arena formal lost the
                    # latter, leaving an exact N-by-W sequence looking like a
                    # rankless scalar to tensor lowering.  Publish the
                    # descriptor receipt on that same formal; no new storage
                    # or reconstructed identity is introduced.
                    row_shape = tuple(
                        (descriptor.column_shapes or ((),))[0]
                    )
                    argument.accounting = {
                        **accounting,
                        "sequence_id": int(descriptor.sequence_id),
                        "sequence_row_shape": row_shape,
                        "program_abi_storage": "span",
                        "program_abi_rank": 1 + len(row_shape),
                        "tensor_metadata_state": "dynamic",
                    }
                length_id = int(descriptor.length_address_id)
                if length_id in existing_ids:
                    continue
                existing_ids.add(length_id)
                region_function.args.append(SSAValue(
                    length_id,
                    dtype="int64",
                    shape=(1,),
                    accounting={
                        "sequence_length_for": int(descriptor.sequence_id),
                        "compiler_frame_storage": control_function.name,
                    },
                ))
                additions.append((int(descriptor.sequence_id), length_id))
            if not additions:
                continue
            for caller in functions.values():
                for caller_block in caller.blocks.values():
                    for call_instruction in caller_block.instrs:
                        if (
                            call_instruction.op not in {"Call", "call"}
                            or str(call_instruction.attributes.get("callee"))
                            != region_function.name
                        ):
                            continue
                        present = {
                            int(value.id) for value in call_instruction.args
                        }
                        appended = [
                            SSAValue(
                                length_id,
                                dtype="int64",
                                shape=(1,),
                                accounting={
                                    "sequence_length_for": sequence_id,
                                    "compiler_frame_storage": (
                                        control_function.name
                                    ),
                                },
                            )
                            for sequence_id, length_id in additions
                            if length_id not in present
                        ]
                        call_instruction.args.extend(appended)
                        if appended:
                            attributes = dict(call_instruction.attributes)
                            if "feed_ids" in attributes:
                                attributes["feed_ids"] = (
                                    *tuple(attributes["feed_ids"]),
                                    *(int(value.id) for value in appended),
                                )
                            if "feed_shapes" in attributes:
                                attributes["feed_shapes"] = (
                                    *tuple(attributes["feed_shapes"]),
                                    *((1,) for _value in appended),
                                )
                            if "feed_dtypes" in attributes:
                                attributes["feed_dtypes"] = (
                                    *tuple(attributes["feed_dtypes"]),
                                    *("int64" for _value in appended),
                                )
                            call_instruction.attributes = attributes
    # Every function this lowering produced -- the control function and its
    # planned regions -- states shapes for the graph COPY it was lowered
    # from (``identity_concordance.shape_scope_of``).
    if shape_scope is not None:
        for _stamped in functions.values():
            _stamped.metadata.setdefault("shape_scope", tuple(shape_scope))
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

        report(
            f"tensor repository linking start; functions={len(module.functions)}"
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
        report(
            "tensor repository linking complete; "
            f"shortfalls={len(tensor_reference_shortfalls)}"
        )
    from .ir_indexing import lower_indexing_to_ssa_addressing

    lower_indexing_to_ssa_addressing(module.functions)
    if tensor_ssa_reference is not None and legalize_aggregate_adapters(module):
        propagate_repository_ssa_call_metadata(module)
    report(
        f"complete; functions={len(module.functions)} "
        f"shortfalls={len(shortfalls) + len(control_shortfalls) + len(tensor_reference_shortfalls)}"
    )
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
    # The control scope of this lowering, numbered by the book, so the
    # region signatures below and the builder's own rows share one scope.
    from .identity_concordance import current_identity_book as _fused_book

    _fused_label, _fused_serial = _fused_book().mint_scope(
        f"control:{control_name}"
    )
    fused_control_scope = f"{control_name}@control:{_fused_serial}"
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
        _post_region_signature(
            _fused_book(), fused_control_scope, None,
            int(region_index), region_signatures[int(region_index)],
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
        expanded_plan_regions = expand_plan_regions(
            hierarchy_plan,
            first_free_value_id=max(used_ids, default=-1) + 1,
            function_scope=control_name,
        )
        for expanded in expanded_plan_regions.values():
            for instruction in expanded:
                used_ids.update(int(value.id) for value in instruction.args)
                if instruction.res is not None:
                    used_ids.add(int(instruction.res.id))
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
            instructions = copy_region_instructions(expanded_plan_regions[
                (str(region.name), int(region.closure_id))
            ])
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
            _post_region_signature(
                _fused_book(), fused_control_scope, None,
                int(region_index), region_signatures[region_index],
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
