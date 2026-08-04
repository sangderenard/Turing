"""Inspection, decoding, and fail-closed execution orchestration.

This module does not claim that vocabulary recognition equals emulation.
Machine effects are supplied by numeric semantic-token handlers; missing
effects stop with a structured result at the exact instruction address.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Sequence

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
    WAITING_EXTERNAL = 6


@dataclass(frozen=True, slots=True)
class MachineExternalReference:
    """A symbolic guest-binary target installed into an import slot."""

    reference_id: int
    target_address: int
    domain: str
    library: str
    symbol: str


@dataclass(frozen=True, slots=True)
class MachineExternalCallRequest:
    """A deterministic system-port request retained in reversible state."""

    request_id: int
    reference: MachineExternalReference
    instruction_address: int
    return_address: int
    arguments: tuple[int, int, int, int]
    stack_pointer: int


@dataclass(frozen=True, slots=True)
class MachineExecutionState:
    pc: int
    registers: tuple[int, ...] = (0,) * 16
    flags: int = 0
    memory: Mapping[int, int] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    call_stack: tuple[int, ...] = ()
    external_requests: tuple[MachineExternalCallRequest, ...] = ()
    steps: int = 0

    REGISTER_NAMES: ClassVar[tuple[str, ...]] = (
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    )

    def register_contents(self) -> Mapping[str, int]:
        """Expose the complete architectural register surface by name."""

        if len(self.registers) != len(self.REGISTER_NAMES):
            raise ValueError(
                f"machine state has {len(self.registers)} registers; "
                f"expected {len(self.REGISTER_NAMES)}"
            )
        values = {
            name: int(value)
            for name, value in zip(self.REGISTER_NAMES, self.registers)
        }
        values.update({
            "rip": int(self.pc),
            "rflags": int(self.flags),
            "steps": int(self.steps),
            "call_depth": len(self.call_stack),
        })
        return MappingProxyType(values)

    def packed_register_words(self) -> tuple[tuple[int, int], ...]:
        """Return lossless low/high u32 words for the WebGPU display ABI."""

        return tuple(
            (value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF)
            for value in self.register_contents().values()
        )


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
MachineExternalTargetResolver = Callable[[int], MachineExternalReference | None]


class MachineExecutionOrchestrator:
    """Schedule decoded instructions while delegating explicit machine effects."""

    def __init__(
        self,
        program,
        *,
        effect_handlers: Mapping[int, MachineEffectHandler] | None = None,
        predicate_handler: MachinePredicateHandler | None = None,
        indirect_target_handler: MachineIndirectTargetHandler | None = None,
        external_target_resolver: MachineExternalTargetResolver | None = None,
    ) -> None:
        self.program = program
        self.effect_handlers = MappingProxyType(dict(effect_handlers or {}))
        self.predicate_handler = predicate_handler
        self.indirect_target_handler = indirect_target_handler
        self.external_target_resolver = external_target_resolver
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
            handler = self.effect_handlers.get(int(semantic))
            handled = handler(advanced, instruction) if handler is not None else advanced
            return MachineExecutionResult(
                MachineExecutionStatus.RUNNING,
                replace(
                    handled,
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
            handler = self.effect_handlers.get(int(semantic))
            handled = handler(advanced, instruction) if handler is not None else advanced
            return MachineExecutionResult(
                MachineExecutionStatus.RUNNING,
                replace(
                    handled,
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
            handler = self.effect_handlers.get(int(semantic))
            handled = handler(advanced, instruction) if handler is not None else advanced
            reference = (
                self.external_target_resolver(int(target))
                if self.external_target_resolver is not None else None
            )
            if reference is not None:
                return_address = (
                    next_pc
                    if semantic is MachineSemanticToken.INDIRECT_CALL
                    else (state.call_stack[-1] if state.call_stack else next_pc)
                )
                request = MachineExternalCallRequest(
                    request_id=handled.steps,
                    reference=reference,
                    instruction_address=instruction.address,
                    return_address=return_address,
                    arguments=(
                        state.registers[1], state.registers[2],
                        state.registers[8], state.registers[9],
                    ),
                    stack_pointer=handled.registers[4],
                )
                waiting = replace(
                    handled,
                    pc=int(target),
                    call_stack=stack,
                    external_requests=(*handled.external_requests, request),
                )
                return MachineExecutionResult(
                    MachineExecutionStatus.WAITING_EXTERNAL,
                    waiting,
                    f"waiting for guest external reference {reference.library}!{reference.symbol}",
                    int(semantic),
                    instruction,
                )
            return MachineExecutionResult(
                MachineExecutionStatus.RUNNING,
                replace(handled, pc=int(target), call_stack=stack),
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
class MachineExecutionEdge:
    """One retained edge in the reversible execution graph."""

    source: MachineExecutionState
    target: MachineExecutionState
    status: MachineExecutionStatus
    instruction: Any | None = None


@dataclass(slots=True)
class ReversibleMachineExecutor:
    """Exact bidirectional journal around the real machine executor.

    Forward movement invokes :class:`MachineExecutionOrchestrator`; backward
    movement restores the complete prior machine state, including memory and
    control stack. Rewinding and executing again creates a new graph branch by
    discarding only the abandoned future of this journal. ``fork`` preserves
    both alternatives as independently runnable execution threads.
    """

    executor: MachineExecutionOrchestrator
    _states: list[MachineExecutionState]
    _edges: list[MachineExecutionEdge]
    _position: int = 0

    @classmethod
    def create(
        cls,
        executor: MachineExecutionOrchestrator,
        state: MachineExecutionState | None = None,
    ) -> "ReversibleMachineExecutor":
        return cls(executor, [state or executor.initial_state()], [])

    @property
    def state(self) -> MachineExecutionState:
        return self._states[self._position]

    @property
    def position(self) -> int:
        return self._position

    @property
    def history_length(self) -> int:
        return len(self._states)

    @property
    def edges(self) -> tuple[MachineExecutionEdge, ...]:
        return tuple(self._edges)

    def step_forward(self) -> MachineExecutionResult:
        source = self.state
        result = self.executor.step(source)
        del self._states[self._position + 1:]
        del self._edges[self._position:]
        self._states.append(result.state)
        self._edges.append(MachineExecutionEdge(
            source, result.state, result.status, result.instruction,
        ))
        self._position += 1
        return result

    def step_backward(self) -> MachineExecutionState:
        if self._position == 0:
            raise IndexError("machine executor is already at its initial state")
        self._position -= 1
        return self.state

    def commit_external_completion(self, state: MachineExecutionState) -> MachineExecutionState:
        """Journal a shell-supplied completion as a reversible graph edge."""

        source = self.state
        del self._states[self._position + 1:]
        del self._edges[self._position:]
        self._states.append(state)
        self._edges.append(MachineExecutionEdge(
            source, state, MachineExecutionStatus.RUNNING, None,
        ))
        self._position += 1
        return state

    def seek_history(self, position: int) -> MachineExecutionState:
        if not 0 <= position < len(self._states):
            raise IndexError("machine execution history position is out of range")
        self._position = int(position)
        return self.state

    def fork(self) -> "ReversibleMachineExecutor":
        return ReversibleMachineExecutor(
            self.executor,
            list(self._states[:self._position + 1]),
            list(self._edges[:self._position]),
            self._position,
        )


@dataclass(frozen=True, slots=True)
class MachineVirtualMulticore:
    """A group of independently reversible binary-execution heads."""

    cores: tuple[ReversibleMachineExecutor, ...]

    @classmethod
    def create(
        cls,
        executor: MachineExecutionOrchestrator,
        *,
        core_count: int,
        initial_states: Sequence[MachineExecutionState] | None = None,
    ) -> "MachineVirtualMulticore":
        if core_count <= 0:
            raise ValueError("virtual multicore requires at least one core")
        states = tuple(initial_states or ())
        if states and len(states) != core_count:
            raise ValueError("initial_states must contain one state per core")
        return cls(tuple(
            ReversibleMachineExecutor.create(
                executor, states[index] if states else None,
            )
            for index in range(core_count)
        ))

    def cycle_forward(self) -> tuple[MachineExecutionResult, ...]:
        """Advance every execution head once across the virtual core barrier."""

        return tuple(core.step_forward() for core in self.cores)

    def cycle_backward(self) -> tuple[MachineExecutionState, ...]:
        """Restore every execution head one cycle across the same barrier."""

        if any(core.position == 0 for core in self.cores):
            raise IndexError("at least one virtual core is already at initial state")
        return tuple(core.step_backward() for core in self.cores)

    def register_contents(self) -> tuple[Mapping[str, int], ...]:
        return tuple(core.state.register_contents() for core in self.cores)

    def packed_register_words(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        return tuple(core.state.packed_register_words() for core in self.cores)


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
    "MachineExternalCallRequest",
    "MachineExternalReference",
    "MachineExternalTargetResolver",
    "MachineExecutionEdge",
    "MachineExecutionResult",
    "MachineExecutionState",
    "MachineExecutionStatus",
    "MachineVirtualMulticore",
    "MachineIndirectTargetHandler",
    "MachineOrchestrationMode",
    "MachineOrchestrationResult",
    "MachinePredicateHandler",
    "ReversibleMachineExecutor",
    "orchestrate_machine_program",
]
