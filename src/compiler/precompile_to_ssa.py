"""Lower an existing Turing numerical precompile and control plan into SSA."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

import networkx as nx

from .control_source import (
    CallBlock,
    ConditionalBlock,
    ControlBlock,
    ControlExpression,
    ControlProgram,
    ControlSequenceMutation,
    LoopControlBlock,
    LoopBlock,
    ParallelDeployment,
    SequenceBlock,
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
    plan_region_to_ssa_instrs,
)
from .precompile_ssa_validator import (
    PrecompileSSAValidationResult,
    ssa_handler_for_precompile,
    validate_precompile_ssa_compatibility,
)
from .ssa_features import XOROSHIRO128SS_FILL, link_required_ssa_features
from ..common.tensors.fused_ir import FusedProgram, Meta, OpStep
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
        },
    )
    if evolution is not None and ssa_evolution is not None:
        evolution.bind_artifact(function, ssa_evolution)
        evolution.close_graph(ssa_evolution)
    return function, tuple(shortfalls)


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
        region_value_meta: Mapping[int, Meta] | None = None,
        value_aliases: Mapping[int, int] | None = None,
        inout_value_ids: tuple[int, ...] = (),
        output_value_ids: tuple[int, ...] = (),
        named_output_histories: Mapping[str, tuple[int, ...]] | None = None,
        value_name_histories: Mapping[str, tuple[int, ...]] | None = None,
        parameter_names: tuple[str, ...] = (),
        sequence_initializations: tuple[tuple[int, str, int], ...] = (),
        sequence_declarations: tuple[tuple[int, str, int, bool], ...] = (),
        sequence_memberships: tuple[tuple[int, int, int, bool], ...] = (),
        table_lookups: tuple[tuple[int, int | tuple[int, ...], int], ...] = (),
        table_stores: tuple[
            tuple[int, int | tuple[int, ...], int, int], ...
        ] = (),
        table_deletions: tuple[
            tuple[int, int | tuple[int, ...], int | None, str], ...
        ] = (),
        retained_sequence_ids: tuple[int, ...] = (),
        nested_sequence_ids: tuple[int, ...] = (),
        nested_row_target_ids: tuple[int, ...] = (),
        selected_nested_sequence_ids: tuple[int, ...] = (),
        variant_projected_target_ids: tuple[int, ...] = (),
        region_array_feed_ids: Mapping[int, tuple[int, ...]] | None = None,
        nested_row_projections: tuple[tuple[int, int, int, str], ...] = (),
        table_region_operations: Mapping[int, tuple[tuple[str, tuple[Any, ...]], ...]] | None = None,
        table_region_post_operations: Mapping[int, tuple[tuple[str, tuple[Any, ...]], ...]] | None = None,
        table_epilogue_operations: tuple[tuple[str, tuple[Any, ...]], ...] = (),
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
        self.region_signatures = dict(region_signatures or {})
        self.arguments: list[SSAValue] = []
        self.external_values: dict[int, SSAValue] = {}
        self.declared_parameter_only_ids: set[int] = set()
        self.sequence_descriptors: dict[int, SSASequenceDescriptor] = {}
        self.sequence_storage_values: dict[int, tuple[SSAValue, ...]] = {}
        self.sequence_status_values: dict[int, SSAValue] = {}
        self.sequence_helper_functions: dict[str, Function] = {}
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
            column_dtypes: list[str | None] = [None] * int(column_count)
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
                column_dtypes=tuple(column_dtypes),
            )
        for sequence_id, policy, column_count in sequence_initializations:
            descriptor_policy = (
                "duplicates" if str(policy).startswith("fill=") else str(policy)
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
                descriptor, function_name=helper_name
            )
            self.sequence_helper_functions.update(
                (function.name, function) for function in lowering.functions
            )
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

        helper_name = f"ssa_sequence_{int(sequence_id)}_lookup"
        lowering = lower_table_lookup(descriptor, function_name=helper_name)
        self.sequence_helper_functions.update(
            (function.name, function) for function in lowering.functions
        )
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
        self.emit(
            Handler.Call,
            [
                *self.sequence_storage_values[int(sequence_id)],
                self.sequence_status_values[int(sequence_id)],
                *self._table_query_values(query_id),
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
        lowering = lower_table_store(descriptor, function_name=helper_name)
        self.sequence_helper_functions.update(
            (function.name, function) for function in lowering.functions
        )
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
                    pool, function_name=helper_name
                )
                self.sequence_helper_functions.update(
                    (function.name, function)
                    for function in lowering.functions
                )
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
                descriptor, function_name=helper_name
            )
        else:
            helper_name = f"ssa_sequence_{int(sequence_id)}_delete"
            lowering = lower_table_delete(descriptor, function_name=helper_name)
        self.sequence_helper_functions.update(
            (function.name, function) for function in lowering.functions
        )
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
        lowering = lower_sequence_fill(descriptor, function_name=helper_name)
        self.sequence_helper_functions.update(
            (function.name, function) for function in lowering.functions
        )
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
            descriptor, function_name=helper_name
        )
        self.sequence_helper_functions.update(
            (function.name, function) for function in lowering.functions
        )
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
        lower_id: int, upper_id: int,
    ) -> None:
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
            destination, source, function_name=helper_name
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
        self.sequence_helper_functions.update(
            (function.name, function) for function in lowering.functions
        )
        storage = tuple({
            value.id: value
            for value in (
                *self.sequence_storage_values[int(destination_id)],
                *self.sequence_storage_values[int(source_id)],
            )
        }.values())
        status = self.fresh_value(dtype="int")
        self.emit(
            Handler.Call,
            [
                *storage,
                self.external_value(int(lower_id), dtype="int"),
                self.external_value(int(upper_id), dtype="int"),
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
            destination, source, function_name=helper_name
        )
        if not lowering.complete:
            self.shortfalls.extend(
                SSALoweringShortfall(
                    "ssa-sequence", item.code.value, self.function_name,
                    item.reason,
                ) for item in lowering.shortfalls
            )
            return
        self.sequence_helper_functions.update(
            (function.name, function) for function in lowering.functions
        )
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
            descriptor, function_name=helper_name
        )
        self.sequence_helper_functions.update(
            (function.name, function) for function in lowering.functions
        )
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
            destination, source, function_name=helper_name
        )
        self.sequence_helper_functions.update(
            (function.name, function) for function in lowering.functions
        )
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
    ) -> SSAValue:
        value_id = int(value_id)
        value = self.external_values.get(value_id)
        if value is not None:
            if value in self.arguments and value_id not in self.inout_value_ids:
                # A preallocated arena is commonly both the initial value
                # entering control and the destination published by a later
                # region.  SSA versions the write; it is not an identity
                # conflict.  The source value ID stays in accounting so the
                # public arena-address policy can rotate the two versions.
                value = self.fresh_value(dtype=dtype)
                value.accounting["source_value_id"] = value_id
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
        result = self.produced_value(result_id)
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

    def emit_region_call(self, region_index: int, *, location: str) -> None:
        if self.emit_table_region_operations(region_index):
            return
        callee = self.region_callees.get(
            region_index,
            f"numerical_region_{region_index}",
        )
        feeds, outputs = self.region_signatures.get(
            region_index, ((), ())
        )
        array_feeds = self.region_array_feed_ids.get(int(region_index), set())
        arguments = [
            self.variant_row_values.get(int(value_id), self.external_value(value_id))
            if int(value_id) in array_feeds
            else self.external_value(value_id)
            for value_id in feeds
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
                "output_ids": outputs,
                "result_convention": "ssa.aggregate",
            },
        )
        if aggregate is not None:
            for output_index, output_id in enumerate(outputs):
                index = self.constant_value(output_index)
                self.indexed_load(
                    aggregate,
                    index,
                    output_id,
                    attributes={
                        "region_index": region_index,
                        "aggregate_index": output_index,
                        "source_output_id": output_id,
                    },
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
                    "dim": 0,
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

    def _lower(self, block: ControlBlock, *, path: str = "root") -> None:
        if isinstance(block, SequenceBlock):
            for index, child in enumerate(block.blocks):
                self.lower(child, path=f"{path}.sequence[{index}]")
            return
        if isinstance(block, StatementBlock):
            for index, line in enumerate(block.lines):
                match = _REGION_MARKER.fullmatch(str(line))
                location = f"{path}.statement[{index}]"
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
            if block.predicate_value_id is None:
                self.branch(target)
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
                self.current = fallthrough
            return
        if isinstance(block, CallBlock):
            # Hierarchy planning has already unified the bound value IDs.
            # CallBlock is lexical organization around the nested compiled
            # control, not an additional runtime invocation.
            self.lower(block.callee, path=f"{path}.callee")
            return
        if isinstance(block, ValidationBlock):
            predicate = self.external_value(
                block.predicate_value_id,
                dtype="bool",
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
                },
            )
            self.branch(passed)
            self.current = passed
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
        column_count: int = 1,
        retains_deleted_rows: bool = False,
        nested_table: bool = False,
        nested_value_dtype: str | None = None,
        element_dtype: str | None = None,
        column_dtypes: tuple[str | None, ...] = (),
    ) -> SSASequenceDescriptor | None:
        value_id = int(value_id)
        key_columns = (
            tuple(range(max(1, int(column_count) - 1)))
            if policy == "unique" else ()
        )
        existing = self.sequence_descriptors.get(value_id)
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
        first_dtype = element_dtype or (
            column_dtypes[0] if column_dtypes else None
        )
        data = self.external_value(value_id, dtype=first_dtype)
        extra_columns = tuple(
            self.fresh_value(dtype=(
                "int"
                if nested_table and index == int(column_count) - 2
                else str(
                    column_dtypes[index + 1]
                    if index + 1 < len(column_dtypes)
                    and column_dtypes[index + 1] is not None
                    else data.dtype or "unknown"
                )
            ))
            for index in range(max(0, int(column_count) - 1))
        )
        self.arguments.extend(extra_columns)
        # Mutable scalar cells are one-element typed arenas at the C ABI, not
        # opaque pointer-typed scalars (which the Fortran emitter would have
        # no element type for and could accidentally pass by value).
        length_address = self.fresh_value(dtype="int", shape=(1,))
        capacity = self.fresh_value(dtype="int")
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
        if not mutation.argument_value_ids:
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

        from .ir_sequence_tables import (
            lower_sequence_add,
            lower_sequence_append,
            lower_sequence_extend,
        )

        call_arguments: tuple[SSAValue, ...]
        if operation in {"append", "add"}:
            if len(mutation.argument_value_ids) != 1:
                self.shortfalls.append(SSALoweringShortfall(
                    "ssa-sequence", operation, location,
                    "row insertion currently requires exactly one value column",
                ))
                return
            function_name = (
                f"ssa_sequence_{destination.sequence_id}_{operation}"
            )
            lowering = (
                lower_sequence_append(destination, function_name=function_name)
                if operation == "append"
                else lower_sequence_add(destination, function_name=function_name)
            )
            call_arguments = (
                *self.sequence_storage_values[destination.sequence_id],
                self.external_value(mutation.argument_value_ids[0]),
            )
        elif operation == "extend":
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
            source = self._sequence_descriptor(
                source_id,
                policy="duplicates",
                writable=False,
                location=location,
            )
            if source is None:
                return
            function_name = (
                f"ssa_sequence_{destination.sequence_id}_extend_"
                f"{source.sequence_id}"
            )
            lowering = lower_sequence_extend(
                destination, source, function_name=function_name
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
        for helper in lowering.functions:
            self.sequence_helper_functions[helper.name] = helper
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

        predicate = (
            self.lower_control_expression(conditional.predicate_expression)
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

        self.current = true_block
        self.lower(conditional.body, path=f"{path}.body")
        true_exit = self.current
        if not true_exit.successors:
            self.branch(merge_block)

        self.current = false_block
        if conditional.orelse is not None:
            self.lower(conditional.orelse, path=f"{path}.orelse")
        false_exit = self.current
        if not false_exit.successors:
            self.branch(merge_block)

        self.current = merge_block
        for (
            true_value_id, false_value_id, initial_value_id, merged_value_id,
        ) in conditional.carried_aliases:
            initial = self.external_value(int(initial_value_id))
            true_value = (
                initial if int(true_value_id) == int(initial_value_id)
                else self.external_value(int(true_value_id), dtype=initial.dtype)
            )
            false_value = (
                initial if int(false_value_id) == int(initial_value_id)
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
                "source_loop_node_id": None,
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
            source = self.external_value(iterable_id)
            if projection is not None:
                column_key = (int(iterable_id), int(projection))
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
                    if int(projection) == 0:
                        source = self.external_value(
                            iterable_id, dtype=target_dtype
                        )
                    else:
                        source = self.fresh_value(dtype=target_dtype)
                        source.accounting.update({
                            "projected_row_source_id": int(iterable_id),
                            "projected_row_column": int(projection),
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
                            "projected_variant_column": int(projection),
                        })
                        self.arguments.append(handle_source)
                        self.variant_handle_columns[column_key] = handle_source
                    scalar_value = self.indexed_load(
                        scalar_source,
                        induction,
                        target_id,
                        attributes={
                            "binding": "projected_variant_scalar",
                            "induction": loop.induction,
                            "projection": projection,
                        },
                    )
                    self.bind_nested_row(
                        handle_source,
                        induction,
                        int(target_id),
                        child_key=(
                            "projected", int(iterable_id), int(projection)
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
                    induction,
                    int(target_id),
                    child_key=(
                        "projected", int(iterable_id), int(projection)
                    ),
                    attributes={
                        "induction": loop.induction,
                        "projection": projection,
                    },
                )
                continue
            self.indexed_load(
                source,
                induction,
                target_id,
                attributes={
                    "binding": "projected_iterable",
                    "induction": loop.induction,
                    "projection": projection,
                },
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
        try:
            self.lower(loop.body, path=f"{path}.body")
            for mutation in loop.sequence_mutations:
                self.lower_sequence_mutation(mutation, path=path)
        finally:
            self.loop_targets.pop()
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
        produced_results = {
            id(instruction.res)
            for basic_block in self.blocks.values()
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
            # On the exit edge the header Phi is the final carried value and
            # dominates every post-loop consumer, including a zero-trip loop.
            self.external_values[initial_id] = current
            self.external_values[updated_id] = current
        self.current = exit_block
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
            self.external_values[initial_id] = current
        self.conditional_branch(current_predicate, body, exit_block)

        self.current = body
        self.loop_targets.append((latch, exit_block))
        try:
            self.lower(loop.body, path=f"{path}.body")
            for mutation in loop.sequence_mutations:
                self.lower_sequence_mutation(mutation, path=path)
        finally:
            self.loop_targets.pop()
        if not self.current.successors:
            self.branch(latch)

        self.current = latch
        self.external_values[int(loop.predicate_value_id)] = next_predicate
        self.lower(loop.condition, path=f"{path}.condition.latch")
        if loop.predicate_expression is not None:
            self.lower_control_expression(
                loop.predicate_expression,
                result_override=next_predicate,
            )
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
        self.current = exit_block

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
        for name, history in self.named_output_histories.items():
            value = next((
                self.external_values[value_id]
                for value_id in reversed(history)
                if value_id in self.external_values
            ), None)
            if value is None:
                continue
            named_returns.append((name, int(value.id)))
            if value.id not in returned_ids:
                returned.append(value)
                returned_ids.add(value.id)
        for value_id in self.output_value_ids:
            value = self.external_values.get(value_id)
            if value is not None and value.id not in returned_ids:
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
        parameter_value_names = tuple(
            (name, value_id)
            for name, value_id in value_names
            if name in self.parameter_names
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
                    "value_names": tuple(value_names),
                    "parameter_names": parameter_value_names,
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
        if self.evolution is not None and self.ssa_evolution is not None:
            self.evolution.bind_artifact(function, self.ssa_evolution)
            self.evolution.close_graph(self.ssa_evolution)
        return function, tuple(self.shortfalls)


def lower_control_program_to_ssa(
    program: ControlProgram,
    *,
    function_name: str = "planned_control",
    first_value_id: int = 0,
    region_callees: dict[int, str] | None = None,
    region_signatures: dict[
        int, tuple[tuple[int, ...], tuple[int, ...]]
    ] | None = None,
    region_value_meta: Mapping[int, Meta] | None = None,
    value_aliases: Mapping[int, int] | None = None,
    inout_value_ids: tuple[int, ...] = (),
    output_value_ids: tuple[int, ...] = (),
    named_output_histories: Mapping[str, tuple[int, ...]] | None = None,
    value_name_histories: Mapping[str, tuple[int, ...]] | None = None,
    parameter_names: tuple[str, ...] = (),
    sequence_initializations: tuple[tuple[int, str, int], ...] = (),
    sequence_declarations: tuple[tuple[int, str, int, bool], ...] = (),
    sequence_memberships: tuple[tuple[int, int, int, bool], ...] = (),
    table_lookups: tuple[tuple[int, int | tuple[int, ...], int], ...] = (),
    table_stores: tuple[
        tuple[int, int | tuple[int, ...], int, int], ...
    ] = (),
    table_deletions: tuple[
        tuple[int, int | tuple[int, ...], int | None, str], ...
    ] = (),
    retained_sequence_ids: tuple[int, ...] = (),
    nested_sequence_ids: tuple[int, ...] = (),
    nested_row_target_ids: tuple[int, ...] = (),
    selected_nested_sequence_ids: tuple[int, ...] = (),
    variant_projected_target_ids: tuple[int, ...] = (),
    region_array_feed_ids: Mapping[int, tuple[int, ...]] | None = None,
    nested_row_projections: tuple[tuple[int, int, int, str], ...] = (),
    table_region_operations: Mapping[int, tuple[tuple[str, tuple[Any, ...]], ...]] | None = None,
    table_region_post_operations: Mapping[int, tuple[tuple[str, tuple[Any, ...]], ...]] | None = None,
    table_epilogue_operations: tuple[tuple[str, tuple[Any, ...]], ...] = (),
) -> tuple[Function, tuple[SSALoweringShortfall, ...]]:
    builder = _ControlSSABuilder(
        program,
        function_name=function_name,
        first_value_id=first_value_id,
        region_callees=region_callees,
        region_signatures=region_signatures,
        region_value_meta=region_value_meta,
        value_aliases=value_aliases,
        inout_value_ids=inout_value_ids,
        output_value_ids=output_value_ids,
        named_output_histories=named_output_histories,
        value_name_histories=value_name_histories,
        parameter_names=parameter_names,
        sequence_initializations=sequence_initializations,
        sequence_declarations=sequence_declarations,
        sequence_memberships=sequence_memberships,
        table_lookups=table_lookups,
        table_stores=table_stores,
        table_deletions=table_deletions,
        retained_sequence_ids=retained_sequence_ids,
        nested_sequence_ids=nested_sequence_ids,
        nested_row_target_ids=nested_row_target_ids,
        selected_nested_sequence_ids=selected_nested_sequence_ids,
        variant_projected_target_ids=variant_projected_target_ids,
        region_array_feed_ids=region_array_feed_ids,
        nested_row_projections=nested_row_projections,
        table_region_operations=table_region_operations,
        table_region_post_operations=table_region_post_operations,
        table_epilogue_operations=table_epilogue_operations,
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
    return builder.finish()


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
) -> Function:
    """Rewrite a method's control function to pass instance state as a slot arena.

    ``self`` becomes a sized field array. Each field read is a ``Load`` from its
    slot (producing the value the body already consumes, so it is no longer a
    free input); each field write is a ``Store`` of its source into the slot. The
    parameter list is rebuilt as ``(self, *non-self params)`` -- the read field
    values leave the signature because the loads now produce them, and ``self``
    joins it as the object arena. A backend that indexes arrays renders the slot
    accesses as ``self(slot + 1)`` and marks a written arena ``intent(inout)``.

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
        return control_function

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

    self_array = SSAValue(int(self_value_id), dtype=dtype, shape=(field_count,))
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

    def slot_address(slot: int) -> tuple[list[Instr], SSAValue]:
        index = SSAValue(fresh(), dtype="int64")
        address = SSAValue(fresh())
        return (
            [
                Instr("Const", [], index, attributes={"value": int(slot)}),
                Instr("GetElementPtr", [self_array, index], address),
            ],
            address,
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
    first_consumer_position: dict[int, int] = {}
    for position, (_name, instruction) in enumerate(flat):
        if instruction.res is not None:
            producer_position.setdefault(int(instruction.res.id), position)
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
        prelude, address = slot_address(slot)
        if kind == "read":
            value_dtype = (
                "opaque_ref" if int(slot) in reference_slots else dtype
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
            source_dtype = "opaque_ref" if reference is not None else dtype
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
                        SSAValue(int(value_id), dtype=dtype),
                        attributes={"value": const_sources[int(value_id)]},
                    )
                )
            group += [
                *prelude,
                Instr(
                    "Store",
                    [SSAValue(int(value_id), dtype=source_dtype), address],
                    None,
                    attributes={
                        "opaque_reference_storage": True,
                        "field_slot": int(slot),
                    } if source_dtype == "opaque_ref" else {},
                ),
            ]
            producer = producer_position.get(int(value_id))
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
        SSAValue(int(value_id), dtype=dtype)
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
    arguments = [self_array]
    seen_argument_ids = {int(self_value_id)}
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
        metadata=dict(control_function.metadata),
    )


def lower_control_sections_to_ssa(
    control: ControlProgram,
    *,
    hierarchy_plan: PlanClosure | None = None,
    control_name: str = "planned_control",
    identity_table: Mapping[str, tuple[int, ...]] | None = None,
    function_outputs: tuple[str, ...] = (),
    function_parameters: tuple[str, ...] = (),
    value_dtypes: Mapping[int, str] | None = None,
    required_output_value_ids: tuple[int, ...] = (),
    region_output_value_ids: Mapping[int, tuple[int, ...]] | None = None,
    record_field_write_value_ids: tuple[int, ...] = (),
    self_value_id: int | None = None,
    field_ops: tuple[tuple[str, int, int], ...] = (),
    field_const_sources: Mapping[int, Any] | None = None,
    field_count: int = 0,
    field_names: tuple[str, ...] = (),
    record_identity: str | None = None,
    sequence_initializations: tuple[tuple[int, str, int], ...] = (),
    field_aliases: tuple[tuple[int, int], ...] = (),
    sequence_declarations: tuple[tuple[int, str, int, bool], ...] = (),
    sequence_memberships: tuple[tuple[int, int, int, bool], ...] = (),
    table_lookups: tuple[tuple[int, int | tuple[int, ...], int], ...] = (),
    table_stores: tuple[
        tuple[int, int | tuple[int, ...], int, int], ...
    ] = (),
    table_deletions: tuple[
        tuple[int, int | tuple[int, ...], int | None, str], ...
    ] = (),
    retained_sequence_ids: tuple[int, ...] = (),
    nested_sequence_ids: tuple[int, ...] = (),
    nested_record_fields: tuple[tuple[int, str, int], ...] = (),
    sequence_augassigns: tuple[tuple[int, int, int], ...] = (),
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
    nested_row_projections: tuple[tuple[int, int, int, str], ...] = (),
    string_table: Any = None,
    tensor_ssa_reference: Any = None,
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
    region_value_meta: dict[int, Meta] = {}
    nested_row_target_ids: set[int] = set()
    selected_nested_sequence_ids: set[int] = set()
    variant_projected_target_ids: set[int] = set()
    region_array_feed_ids: dict[int, set[int]] = {}
    section_outputs: dict[str, tuple[SSAValue, ...]] = {}
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
            int(destination_id), int(source_id), int(lower_id), int(upper_id)
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
        region_value_meta[int(lower_id)] = Meta((), "int")
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
            planned_instructions = plan_region_to_ssa_instrs(planned)
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
        variant_projected_target_ids.update(
            target_id
            for target_id in nested_row_target_ids
            if target_id in scalar_use_ids
        )
        # The same heterogeneous contract is required when source pursuit has
        # specialized away the owning loop and left its payload as a direct
        # method input.  Its uses, not its spelling, prove the two columns.
        variant_projected_target_ids.update(
            (indexed_base_ids & scalar_use_ids) - declared_sequence_ids
        )
        for planned in hierarchy_plan.items:
            if not (
                isinstance(planned, PlanClosure)
                and planned.name.startswith("region_")
            ):
                continue
            region_index = int(planned.name.rsplit("_", 1)[1])
            instructions = plan_region_to_ssa_instrs(planned)
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
                () if current is None else tuple(current.shape), "int"
            )
    table_region_operations: dict[
        int, list[tuple[str, tuple[Any, ...]]]
    ] = {}
    table_region_post_operations: dict[
        int, list[tuple[str, tuple[Any, ...]]]
    ] = {}
    region_value_aliases: dict[int, int] = {}
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
            for instruction in plan_region_to_ssa_instrs(item)
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

    def collect_resolved_plan_dependencies(closure: PlanClosure) -> None:
        """Collect the dependency boundary already resolved by the planner."""

        resolved_plan_live_value_ids.update(int(value_id) for value_id in closure.captures)
        for item in closure.items:
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
        for planned in hierarchy_plan.items:
            if not (
                isinstance(planned, PlanClosure)
                and planned.name.startswith("region_")
            ):
                continue
            planned_index = int(planned.name.rsplit("_", 1)[1])
            planned_instructions = plan_region_to_ssa_instrs(planned)
            planned_region_instructions[planned_index] = planned_instructions

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
            # Namespace regions by their owning method so two methods that each
            # carve a ``region_0`` do not collide in one shared library, and so
            # the control call the lowering emits already targets this symbol.
            region_name = f"{control_name}__planned_region_{region_index}"
            instructions = list(plan_region_to_ssa_instrs(region))
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
                    or value_id in consumed_after_field_resolution
                    and value_id not in produced_after_field_resolution
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
                        tuple(map(int, authoritative.shape))
                        if authoritative is not None
                        else tuple(semantic.shape)
                        if semantic is not None and semantic.shape
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

            # One SSA identity has one contract inside the function.  Rebind
            # the instruction graph itself, not only its formal arguments and
            # outputs, so target inference cannot see a stale float result and
            # an integer function output for the same value ID.
            for instruction in instructions:
                instruction.args = [
                    typed_region_value(value.id) for value in instruction.args
                ]
                if instruction.res is not None:
                    instruction.res = typed_region_value(instruction.res.id)

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
                for value_id in effective_captures
                if (
                    value_id in region_inout_ids
                    or value_id in consumed and value_id not in produced
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
            outputs = tuple(sorted(produced & required_outputs))
            outputs = tuple(
                value_id for value_id in outputs
                if value_id not in region_value_aliases
            )
            # The region's formal parameters are its captures only. Its outputs
            # are declared as ``intent(out)`` dummies by the target from the
            # ``outputs`` map (returned below as ``section_outputs``), exactly as
            # the fused numerical region path does -- never by placing them in
            # ``args``, which would misread an output as an in/out alias.
            arguments = [typed_region_value(vid) for vid in effective_captures]
            region_function = Function(
                region_name,
                arguments,
                {"entry": BasicBlock("entry", instructions)},
            )
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
    for value_id, value_dtype in (value_dtypes or {}).items():
        value_id = int(value_id)
        existing = region_value_meta.get(value_id)
        region_value_meta[value_id] = Meta(
            tuple(existing.shape or ()) if existing is not None else (),
            str(value_dtype),
            existing.device if existing is not None else None,
        )
    control_function, control_shortfalls = lower_control_program_to_ssa(
        control,
        function_name=control_name,
        first_value_id=max(graph_value_ids) + 1,
        region_callees=region_callees,
        region_signatures=region_signatures,
        region_value_meta=region_value_meta,
        value_aliases=region_value_aliases,
        inout_value_ids=tuple(map(int, record_field_write_value_ids)),
        named_output_histories={
            str(name): tuple(map(int, (identity_table or {}).get(name, ())))
            for name in function_outputs
        },
        value_name_histories=identity_table,
        parameter_names=function_parameters,
        sequence_initializations=sequence_initializations,
        sequence_declarations=sequence_declarations,
        sequence_memberships=sequence_memberships,
        table_lookups=table_lookups,
        table_stores=table_stores,
        table_deletions=table_deletions,
        retained_sequence_ids=retained_sequence_ids,
        nested_sequence_ids=nested_sequence_ids,
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
    scalar_slots = tuple(
        slot for slot in range(int(field_count))
        if slot not in sequence_field_slots
    )
    compact_slot = {slot: index for index, slot in enumerate(scalar_slots)}
    scalar_field_ops = tuple(
        (kind, value_id, compact_slot[slot])
        for kind, value_id, slot in field_ops
        if slot in compact_slot
    )
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
        control_function = _inject_field_slot_access(
            control_function,
            self_value_id=int(self_value_id),
            non_self_param_ids=non_self_param_ids,
            field_ops=scalar_field_ops,
            field_const_sources=field_const_sources or {},
            field_count=len(scalar_slots),
            output_value_ids=output_value_ids,
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
    # Lower subscript ops (Indexed/IndexedStore) to the universal address
    # vocabulary (GetElementPtr + Load/Store) that every backend already speaks,
    # so the subscript lowering lives once at the SSA level, not per backend.
    from .ir_indexing import lower_indexing_to_ssa_addressing

    lower_indexing_to_ssa_addressing(functions)
    # Tokenize every string constant to its universal fnv1a token before
    # emission, so a word is a 64-bit value the target expresses like any other
    # constant instead of an inexpressible literal.
    from .ir_string_interning import tokenize_ssa_string_constants

    tokenize_ssa_string_constants(functions, string_table)
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
                record_fields.append(SSARecordFieldDescriptor(
                    name,
                    (
                        SSARecordFieldStorage.REFERENCE
                        if old_slot in reference_slots
                        else SSARecordFieldStorage.SCALAR
                    ),
                    storage_identity=storage_identity,
                    value_ids=(int(self_value_id),),
                    offset=compact_slot[old_slot],
                    dtype=("opaque_ref" if old_slot in reference_slots else None),
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
        from .tensor_ssa_lowering import lower_tensor_calls_to_repository_ssa

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
    return module, tuple((
        *shortfalls,
        *control_shortfalls,
        *tensor_reference_shortfalls,
    )), section_outputs


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
    # Structural subscripts are valid SSA vocabulary during composition, then
    # become the universal address primitives every backend already consumes.
    from .ir_indexing import lower_indexing_to_ssa_addressing

    lower_indexing_to_ssa_addressing(functions)
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
        from .tensor_ssa_lowering import lower_tensor_calls_to_repository_ssa

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
