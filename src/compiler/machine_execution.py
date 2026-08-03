"""Inspection, decoding, and fail-closed execution orchestration.

This module does not claim that vocabulary recognition equals emulation.
Machine effects are supplied by numeric semantic-token handlers; missing
effects stop with a structured result at the exact instruction address.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .binary_structure_graph import (
    BinaryStructureGraph,
    build_pe_binary_structure_graph,
)
from .machine_reference_vocabulary import (
    MachineSemanticToken,
    RelativeAddressOperand,
)


class MachineOrchestrationMode(IntEnum):
    INSPECT = 0
    DECODE = 1
    EMULATE = 2


class MachineExecutionStatus(IntEnum):
    RUNNING = 0
    HALTED = 1
    BLOCKED_EFFECT = 2
    BLOCKED_CONTROL = 3
    TRAPPED = 4
    STEP_LIMIT = 5


@dataclass(frozen=True, slots=True)
class MachineExecutionState:
    pc: int
    registers: tuple[int, ...] = (0,) * 16
    flags: int = 0
    memory: Mapping[int, int] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    call_stack: tuple[int, ...] = ()
    steps: int = 0


@dataclass(frozen=True, slots=True)
class MachineExecutionResult:
    status: MachineExecutionStatus
    state: MachineExecutionState
    reason: str = ""
    missing_semantic_token: int | None = None
    instruction: Any | None = None


MachineEffectHandler = Callable[[MachineExecutionState, Any], MachineExecutionState]
MachinePredicateHandler = Callable[[MachineExecutionState, Any], bool]
MachineIndirectTargetHandler = Callable[[MachineExecutionState, Any], int]


class MachineExecutionOrchestrator:
    """Schedule decoded instructions while delegating explicit machine effects."""

    def __init__(
        self,
        program,
        *,
        effect_handlers: Mapping[int, MachineEffectHandler] | None = None,
        predicate_handler: MachinePredicateHandler | None = None,
        indirect_target_handler: MachineIndirectTargetHandler | None = None,
    ) -> None:
        self.program = program
        self.effect_handlers = MappingProxyType(dict(effect_handlers or {}))
        self.predicate_handler = predicate_handler
        self.indirect_target_handler = indirect_target_handler
        instructions: dict[int, Any] = {}
        for record in program.functions:
            for instruction in record.report.instructions:
                previous = instructions.setdefault(instruction.address, instruction)
                if previous is not instruction and previous != instruction:
                    raise ValueError(
                        f"conflicting decoded instructions at {instruction.address:#x}"
                    )
        self.instructions = MappingProxyType(instructions)

    def initial_state(self) -> MachineExecutionState:
        return MachineExecutionState(
            pc=self.program.image.image_base + self.program.image.entrypoint_rva,
        )

    @staticmethod
    def _relative_target(instruction: Any) -> int | None:
        return next((
            operand.target_address
            for operand in instruction.operands
            if isinstance(operand, RelativeAddressOperand)
        ), None)

    def step(self, state: MachineExecutionState) -> MachineExecutionResult:
        instruction = self.instructions.get(state.pc)
        if instruction is None:
            return MachineExecutionResult(
                MachineExecutionStatus.BLOCKED_CONTROL,
                state,
                f"no decoded instruction at program counter {state.pc:#x}",
            )
        next_pc = instruction.address + len(instruction.encoded)
        semantic = instruction.semantic
        advanced = replace(state, pc=next_pc, steps=state.steps + 1)

        if semantic is MachineSemanticToken.RETURN:
            if not state.call_stack:
                return MachineExecutionResult(
                    MachineExecutionStatus.HALTED, advanced,
                    "returned from the outermost emulated frame", instruction=instruction,
                )
            return MachineExecutionResult(
                MachineExecutionStatus.RUNNING,
                replace(
                    advanced,
                    pc=state.call_stack[-1],
                    call_stack=state.call_stack[:-1],
                ),
                instruction=instruction,
            )
        if semantic is MachineSemanticToken.DIRECT_RELATIVE_JUMP:
            target = self._relative_target(instruction)
            if target is None:
                return MachineExecutionResult(
                    MachineExecutionStatus.BLOCKED_CONTROL, state,
                    "direct jump has no relative target", instruction=instruction,
                )
            return MachineExecutionResult(
                MachineExecutionStatus.RUNNING,
                replace(advanced, pc=target),
                instruction=instruction,
            )
        if semantic is MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP:
            if self.predicate_handler is None:
                return MachineExecutionResult(
                    MachineExecutionStatus.BLOCKED_EFFECT, state,
                    "conditional control requires an explicit flags predicate handler",
                    int(semantic), instruction,
                )
            target = self._relative_target(instruction)
            taken = self.predicate_handler(state, instruction)
            return MachineExecutionResult(
                MachineExecutionStatus.RUNNING,
                replace(advanced, pc=target if taken and target is not None else next_pc),
                instruction=instruction,
            )
        if semantic is MachineSemanticToken.DIRECT_RELATIVE_CALL:
            target = self._relative_target(instruction)
            if target is None or target not in self.instructions:
                return MachineExecutionResult(
                    MachineExecutionStatus.BLOCKED_CONTROL, state,
                    "direct call target needs an emulated or host-bound implementation",
                    int(semantic), instruction,
                )
            return MachineExecutionResult(
                MachineExecutionStatus.RUNNING,
                replace(
                    advanced,
                    pc=target,
                    call_stack=(*state.call_stack, next_pc),
                ),
                instruction=instruction,
            )
        if semantic in {
            MachineSemanticToken.INDIRECT_CALL,
            MachineSemanticToken.INDIRECT_JUMP,
        }:
            if self.indirect_target_handler is None:
                return MachineExecutionResult(
                    MachineExecutionStatus.BLOCKED_CONTROL, state,
                    "indirect control requires an explicit target resolver",
                    int(semantic), instruction,
                )
            target = self.indirect_target_handler(state, instruction)
            stack = (
                (*state.call_stack, next_pc)
                if semantic is MachineSemanticToken.INDIRECT_CALL
                else state.call_stack
            )
            return MachineExecutionResult(
                MachineExecutionStatus.RUNNING,
                replace(advanced, pc=int(target), call_stack=stack),
                instruction=instruction,
            )
        if semantic in {
            MachineSemanticToken.BREAKPOINT_TRAP,
            MachineSemanticToken.SOFTWARE_INTERRUPT,
        }:
            return MachineExecutionResult(
                MachineExecutionStatus.TRAPPED,
                advanced,
                f"machine trap semantic {semantic.name}",
                int(semantic), instruction,
            )

        handler = self.effect_handlers.get(int(semantic))
        if handler is None:
            return MachineExecutionResult(
                MachineExecutionStatus.BLOCKED_EFFECT,
                state,
                f"no emulation handler for semantic token {int(semantic)} ({semantic.name})",
                int(semantic), instruction,
            )
        handled = handler(advanced, instruction)
        return MachineExecutionResult(
            MachineExecutionStatus.RUNNING,
            handled,
            instruction=instruction,
        )

    def run(
        self,
        state: MachineExecutionState | None = None,
        *,
        maximum_steps: int = 1_000_000,
    ) -> MachineExecutionResult:
        if maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        active = state or self.initial_state()
        for _ in range(maximum_steps):
            result = self.step(active)
            if result.status is not MachineExecutionStatus.RUNNING:
                return result
            active = result.state
        return MachineExecutionResult(
            MachineExecutionStatus.STEP_LIMIT,
            active,
            f"emulation exceeded {maximum_steps} instructions",
        )


@dataclass(frozen=True, slots=True)
class MachineOrchestrationResult:
    mode: MachineOrchestrationMode
    program: Any
    structure: BinaryStructureGraph | None = None
    execution: MachineExecutionResult | None = None


def orchestrate_machine_program(
    program,
    *,
    mode: MachineOrchestrationMode,
    executor: MachineExecutionOrchestrator | None = None,
    maximum_steps: int = 1_000_000,
) -> MachineOrchestrationResult:
    if mode is MachineOrchestrationMode.INSPECT:
        return MachineOrchestrationResult(
            mode, program, structure=build_pe_binary_structure_graph(program),
        )
    if mode is MachineOrchestrationMode.DECODE:
        return MachineOrchestrationResult(mode, program)
    active = executor or MachineExecutionOrchestrator(program)
    return MachineOrchestrationResult(
        mode, program, execution=active.run(maximum_steps=maximum_steps),
    )


__all__ = [
    "MachineEffectHandler",
    "MachineExecutionOrchestrator",
    "MachineExecutionResult",
    "MachineExecutionState",
    "MachineExecutionStatus",
    "MachineIndirectTargetHandler",
    "MachineOrchestrationMode",
    "MachineOrchestrationResult",
    "MachinePredicateHandler",
    "orchestrate_machine_program",
]
