"""Lower an existing Turing numerical precompile and control plan into SSA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import networkx as nx

from .control_source import (
    CallBlock,
    ControlBlock,
    ControlExpression,
    ControlProgram,
    LoopControlBlock,
    LoopBlock,
    ParallelDeployment,
    SequenceBlock,
    StateMachineTick,
    StatementBlock,
    StreamPublishBlock,
    ValidationBlock,
    WhileBlock,
)
from .deployment_frame import DeploymentJoin
from .hierarchical_plan import (
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
    SSADeploymentLane,
    SSADeploymentRegion,
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
        output_value_ids: tuple[int, ...] = (),
        named_output_histories: Mapping[str, tuple[int, ...]] | None = None,
        value_name_histories: Mapping[str, tuple[int, ...]] | None = None,
        parameter_names: tuple[str, ...] = (),
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
        self.current = self.new_block("entry")

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
        value = self.external_values.get(value_id)
        if value is None:
            value = self._value_from_meta(value_id, dtype=dtype)
            self.external_values[value_id] = value
            self.arguments.append(value)
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
        return _ssa_value(int(value_id), meta)

    def produced_value(
        self,
        value_id: int,
        *,
        dtype: str | None = None,
    ) -> SSAValue:
        value_id = int(value_id)
        value = self.external_values.get(value_id)
        if value is not None:
            if value in self.arguments:
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

    def emit_region_call(self, region_index: int, *, location: str) -> None:
        callee = self.region_callees.get(
            region_index,
            f"numerical_region_{region_index}",
        )
        feeds, outputs = self.region_signatures.get(
            region_index, ((), ())
        )
        arguments = [self.external_value(value_id) for value_id in feeds]
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
        if aggregate is None:
            return
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
                attributes={"block": self.current.name},
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
            self.external_values[initial_id] = current_value
        condition = self.fresh_value(dtype="bool")
        self.emit(Handler.Lt, [induction, stop], condition)
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
        for iterable_id, target_id, induction_name in (
            self.program.iterable_bindings
        ):
            if induction_name != loop.induction:
                continue
            restored_values[int(target_id)] = self.external_values.get(
                int(target_id)
            )
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
            restored_values[int(target_id)] = self.external_values.get(
                int(target_id)
            )
            if projection == "induction":
                self.external_values[int(target_id)] = induction
                continue
            indices = [induction]
            if projection is not None:
                indices.append(self.constant_value(int(projection)))
            self.indexed_load(
                self.external_value(iterable_id),
                indices,
                target_id,
                attributes={
                    "binding": "projected_iterable",
                    "induction": loop.induction,
                    "projection": projection,
                },
            )
        for iterable_id, target_id, induction_name, values in (
            self.program.static_iterable_bindings
        ):
            if induction_name != loop.induction:
                continue
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
        finally:
            self.loop_targets.pop()
        produced_results = {
            id(instruction.res)
            for basic_block in self.blocks.values()
            for instruction in basic_block.instrs
            if instruction.res is not None
        }
        for updated_id, _initial_id, _initial, updated, _current in carried:
            if id(updated) not in produced_results:
                self.shortfalls.append(
                    SSALoweringShortfall(
                        "control",
                        "loop_carried",
                        f"{path}.body",
                        f"carried update value {updated_id} has no producer "
                        "inside the loop body",
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
            if value is not None:
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
    output_value_ids: tuple[int, ...] = (),
    named_output_histories: Mapping[str, tuple[int, ...]] | None = None,
    value_name_histories: Mapping[str, tuple[int, ...]] | None = None,
    parameter_names: tuple[str, ...] = (),
) -> tuple[Function, tuple[SSALoweringShortfall, ...]]:
    builder = _ControlSSABuilder(
        program,
        function_name=function_name,
        first_value_id=first_value_id,
        region_callees=region_callees,
        region_signatures=region_signatures,
        region_value_meta=region_value_meta,
        output_value_ids=output_value_ids,
        named_output_histories=named_output_histories,
        value_name_histories=value_name_histories,
        parameter_names=parameter_names,
    )
    builder.lower(program.root)
    return builder.finish()


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
    field_reads: tuple[tuple[int, int], ...],
    field_writes: tuple[tuple[int, int], ...],
    field_count: int,
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
    next_id = max(existing_ids, default=-1) + 1

    def fresh() -> int:
        nonlocal next_id
        value_id = next_id
        next_id += 1
        return value_id

    self_array = SSAValue(int(self_value_id), dtype=dtype, shape=(field_count,))

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

    # (insert-position, order-within-position, home-block, instructions)
    insertions: list[tuple[int, int, str, list[Instr]]] = []
    field_read_ids: set[int] = set()
    for result_id, slot in field_reads:
        prelude, address = slot_address(slot)
        group = [*prelude, Instr("Load", [address], SSAValue(int(result_id), dtype=dtype))]
        field_read_ids.add(int(result_id))
        position = first_consumer_position.get(int(result_id))
        if position is None:
            continue  # a field read nothing consumes has no place and no effect
        insertions.append((position, 0, flat[position][0], group))
    for source_id, slot in field_writes:
        prelude, address = slot_address(slot)
        group = [*prelude, Instr("Store", [SSAValue(int(source_id), dtype=dtype), address], None)]
        producer = producer_position.get(int(source_id))
        # After the producer; a parameter source has none, so at the top.
        position = producer + 1 if producer is not None else 0
        home = flat[position][0] if position < len(flat) else (
            flat[-1][0] if flat else entry_name
        )
        insertions.append((position, 1, home, group))

    from collections import defaultdict

    inserts_at: dict[int, list[list[Instr]]] = defaultdict(list)
    for position, _order, _home, group in sorted(
        insertions, key=lambda item: (item[0], item[1])
    ):
        inserts_at[position].append(group)

    rebuilt: dict[str, list[Instr]] = {
        name: [] for name in control_function.blocks
    }
    for position, (name, instruction) in enumerate(flat):
        for group in inserts_at.get(position, ()):
            rebuilt[name].extend(group)
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
    for param_id in non_self_param_ids:
        if int(param_id) not in field_read_ids:
            arguments.append(SSAValue(int(param_id), dtype=dtype))

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
    self_value_id: int | None = None,
    field_reads: tuple[tuple[int, int], ...] = (),
    field_writes: tuple[tuple[int, int], ...] = (),
    field_count: int = 0,
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
    section_outputs: dict[str, tuple[SSAValue, ...]] = {}
    shortfalls: list[SSALoweringShortfall] = []
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
            produced = {int(instr.res.id) for instr in instructions}
            consumed = {
                int(argument.id)
                for instr in instructions
                for argument in instr.args
            }
            outputs = tuple(sorted(produced - consumed)) or tuple(
                sorted(produced)
            )
            # The region's formal parameters are its captures only. Its outputs
            # are declared as ``intent(out)`` dummies by the target from the
            # ``outputs`` map (returned below as ``section_outputs``), exactly as
            # the fused numerical region path does -- never by placing them in
            # ``args``, which would misread an output as an in/out alias.
            arguments = [SSAValue(int(vid)) for vid in region.captures]
            functions[region_name] = Function(
                region_name,
                arguments,
                {"entry": BasicBlock("entry", instructions)},
            )
            region_callees[region_index] = region_name
            region_signatures[region_index] = (
                tuple(int(vid) for vid in region.captures),
                outputs,
            )
            section_outputs[region_name] = tuple(
                SSAValue(int(vid)) for vid in outputs
            )
            known_operations = {handler.value for handler in Handler}
            shortfalls.extend(
                SSALoweringShortfall(
                    "planned-region",
                    str(instr.op),
                    f"{region_name}:entry",
                    "operator has no repository SSA handler",
                )
                for instr in instructions
                if str(instr.op) not in known_operations
            )
    control_function, control_shortfalls = lower_control_program_to_ssa(
        control,
        function_name=control_name,
        region_callees=region_callees,
        region_signatures=region_signatures,
        named_output_histories={
            str(name): tuple(map(int, (identity_table or {}).get(name, ())))
            for name in function_outputs
        },
        value_name_histories=identity_table,
        parameter_names=function_parameters,
    )
    if self_value_id is not None and (field_reads or field_writes):
        non_self_param_ids = tuple(
            int((identity_table or {}).get(name, (None,))[-1])
            for name in function_parameters
            if name != "self" and (identity_table or {}).get(name)
        )
        control_function = _inject_field_slot_access(
            control_function,
            self_value_id=int(self_value_id),
            non_self_param_ids=non_self_param_ids,
            field_reads=field_reads,
            field_writes=field_writes,
            field_count=int(field_count),
        )
    functions[control_function.name] = control_function
    module = IRModule(
        link_required_ssa_features(functions),
        recursion_table={
            name: dict(function.metadata.get("recursion_table", {}))
            for name, function in functions.items()
            if function.metadata.get("recursion_table")
        },
    )
    return module, tuple((*shortfalls, *control_shortfalls)), section_outputs


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
    algorithm_import = import_llvm_to_repository_ssa(LLVM_SSA_MODULE)
    functions = dict(algorithm_import.module.functions)
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
                int(instruction.res.id) for instruction in instructions
            }
            consumed = {
                int(argument.id)
                for instruction in instructions
                for argument in instruction.args
            }
            outputs = tuple(sorted(produced - consumed)) or tuple(
                sorted(produced)
            )
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
    functions[control_function.name] = control_function
    module = IRModule(
        link_required_ssa_features(functions),
        recursion_table={
            name: dict(function.metadata.get("recursion_table", {}))
            for name, function in functions.items()
            if function.metadata.get("recursion_table")
        },
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
            *(
                SSALoweringShortfall(
                    "llvm",
                    item.opcode,
                    f"{item.function}:{item.block}",
                    item.reason,
                )
                for item in algorithm_import.shortfalls
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
    "ssa_module_dictionary",
]
