"""Lower an existing Turing numerical precompile and control plan into SSA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import networkx as nx

from .control_source import (
    CallBlock,
    ControlBlock,
    ControlProgram,
    LoopBlock,
    ParallelDeployment,
    SequenceBlock,
    StateMachineTick,
    StatementBlock,
    StreamPublishBlock,
    ValidationBlock,
)
from .precompile_ssa_validator import (
    PrecompileSSAValidationResult,
    ssa_handler_for_precompile,
    validate_precompile_ssa_compatibility,
)
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
    return SSAValue(
        int(value_id),
        dtype=None if meta is None else meta.dtype,
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
    """Translate only operations already named by the existing SSA registry."""

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

    def element_count(value_id: int) -> int:
        meta = metadata.get(value_id)
        count = 1
        for extent in (() if meta is None else (meta.shape or ())):
            count *= int(extent)
        return count

    def tensor_algorithm(step: OpStep) -> str | None:
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
    return (
        Function(
            function_name,
            arguments,
            {
                "entry": BasicBlock(
                    "entry",
                    instructions,
                    [],
                )
            },
        ),
        tuple(shortfalls),
    )


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
    ):
        self.program = program
        self.region_value_meta = dict(region_value_meta or {})
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
                self.shortfalls.append(
                    SSALoweringShortfall(
                        "control",
                        "producer_identity",
                        self.current.name,
                        f"value {value_id} is both a control argument and "
                        "a scheduled-region result",
                    )
                )
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
        index: SSAValue,
        result_id: int,
        *,
        attributes: dict[str, Any],
    ) -> SSAValue:
        address = self.fresh_value(dtype="ptr")
        self.emit(
            Handler.GetElementPtr,
            [source, index],
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
        self.current.instrs.append(
            Instr(
                op.value,
                args,
                result,
                attributes=dict(attributes or {}),
            )
        )

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
        spelling = str(expression)
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

    def lower(self, block: ControlBlock, *, path: str = "root") -> None:
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
                    self.emit_region_call(
                        region_index,
                        location=location,
                    )
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
            # The lanes are independent.  A linear SSA listing is a valid
            # schedule of that partial order; a target may re-parallelize it
            # from the retained lane structure before this final lowering.
            for index, lane in enumerate(block.lanes):
                self.lower(lane, path=f"{path}.lane[{index}]")
            return
        raise TypeError(f"unknown control block {type(block).__name__}")

    def lower_loop(self, loop: LoopBlock, *, path: str) -> None:
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
                },
            )
            self.external_values[initial_id] = current_value
        condition = self.fresh_value(dtype="bool")
        self.emit(Handler.Lt, [induction, stop], condition)
        self.conditional_branch(condition, body, exit_block)

        self.current = body
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
        self.lower(loop.body, path=f"{path}.body")
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
        if not self.current.successors:
            self.branch(latch)

        self.current = latch
        self.emit(
            Handler.Add,
            [induction, step],
            next_induction,
        )
        self.branch(header)
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
        if not self.current.successors:
            self.branch(merge)
        self.current = merge

    def finish(self) -> tuple[Function, tuple[SSALoweringShortfall, ...]]:
        if not self.current.successors and (
            not self.current.instrs
            or self.current.instrs[-1].op
            not in {Handler.Br.value, Handler.CondBr.value, Handler.Ret.value}
        ):
            self.emit(Handler.Ret, [])
        return (
            Function(
                self.function_name,
                self.arguments,
                self.blocks,
            ),
            tuple(self.shortfalls),
        )


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
) -> tuple[Function, tuple[SSALoweringShortfall, ...]]:
    builder = _ControlSSABuilder(
        program,
        function_name=function_name,
        first_value_id=first_value_id,
        region_callees=region_callees,
        region_signatures=region_signatures,
        region_value_meta=region_value_meta,
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
    return IRModule({function.name: function for function in functions})


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


def lower_precompile_and_control_to_ssa(
    artifact: Any,
    control: ControlProgram,
    *,
    numerical_name: str = "numerical_precompile",
    control_name: str = "planned_control",
    region_programs: dict[int, Any] | None = None,
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
    )
    functions[control_function.name] = control_function
    module = IRModule(functions)
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
    "lower_fused_program_to_ssa",
    "lower_precompile_and_control_to_ssa",
    "ssa_module_dictionary",
]
