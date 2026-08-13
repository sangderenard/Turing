"""Inspection, decoding, and fail-closed execution orchestration.

This module does not claim that vocabulary recognition equals emulation.
Machine effects are supplied by numeric semantic-token handlers; missing
effects stop with a structured result at the exact instruction address.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Sequence

from .virtual_filesystem import VirtualFileEffect, VirtualFileSystemState
from .virtual_registry import VirtualRegistryEffect, VirtualRegistryState
from .virtual_memory import VirtualMemoryEffect, VirtualMemoryState

from .binary_structure_graph import (
    BinaryStructureGraph,
    build_pe_binary_structure_graph,
)
from .machine_reference_vocabulary import (
    MachineSemanticToken,
    RelativeAddressOperand,
    VocabularyDecodeError,
    X86ReferenceDecoder,
)


MASK64 = (1 << 64) - 1
MACHINE_TERMINATION_RETURN = MASK64 - 1
MACHINE_LOADER_CALLBACK_RETURN = MASK64 - 2


def _shared_system_key(key: object) -> bool:
    name = str(key)
    if name.startswith(("machine.code_page.", "windows.handle.")):
        return True
    if name in {"windows.thread.next_id", "windows.thread.count"}:
        return True
    if name.startswith("windows.thread."):
        parts = name.split(".")
        return len(parts) > 2 and parts[2].isdigit()
    return False


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
    stack_arguments: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class MachineExternalMemoryWrite:
    """One capability-approved guest-memory effect returned by a shell."""

    address: int
    data: bytes

    def __post_init__(self) -> None:
        if self.address < 0:
            raise ValueError("external memory-write address cannot be negative")
        object.__setattr__(self, "data", bytes(self.data))


@dataclass(frozen=True, slots=True)
class MachineExternalStateWrite:
    """One reversible key/value mutation of the virtual system environment."""

    key: str
    value: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("external state-write key cannot be empty")


@dataclass(frozen=True, slots=True)
class MachineExternalRegisterWrite:
    """One capability-approved write to the contiguous integer register bank."""

    register: int
    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.register < 16:
            raise ValueError("external register-write index must be in [0, 16)")


@dataclass(frozen=True, slots=True)
class MachineExternalEnvironmentWrite:
    """Set or delete one case-insensitive guest environment variable."""

    key: str
    value: str | None

    def __post_init__(self) -> None:
        # Windows reserves leading-'=' names (for example '=C:') for each
        # drive's current directory. Other embedded '=' characters remain
        # invalid separators.
        if (
            not self.key or self.key == "=" or "=" in self.key[1:]
            or "\x00" in self.key
        ):
            raise ValueError("invalid guest environment-variable name")


@dataclass(frozen=True, slots=True)
class MachineExternalTextStateWrite:
    """One reversible textual process-runtime setting."""

    key: str
    value: str

    def __post_init__(self) -> None:
        if not self.key or "\x00" in self.key or "\x00" in self.value:
            raise ValueError("invalid textual system-state effect")


@dataclass(frozen=True, slots=True)
class MachineExternalDeviceWrite:
    """Append or replace bytes in a reversible virtual device buffer."""

    device: str
    data: bytes
    append: bool = True

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("external device write needs a device name")
        object.__setattr__(self, "data", bytes(self.data))


@dataclass(frozen=True, slots=True)
class MachineExternalControlTransfer:
    """Replace control/shadow-stack state for a capability-owned nonlocal jump."""

    address: int
    call_stack: tuple[int, ...]
    vector_registers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.address <= 0:
            raise ValueError("external control transfer needs a positive target")
        object.__setattr__(self, "call_stack", tuple(int(item) for item in self.call_stack))
        if self.vector_registers is not None:
            vectors = tuple(int(item) for item in self.vector_registers)
            if len(vectors) != 16:
                raise ValueError("external control transfer needs all sixteen XMM registers")
            object.__setattr__(self, "vector_registers", vectors)


@dataclass(frozen=True, slots=True)
class MachineExternalDeployment:
    """Durable identity of one child executor selected by a system port."""

    deployment_id: int
    kind: str
    requested_reference: str
    resolved_reference: str
    executor_reference: str
    exit_code: int
    execution_units: int = 0
    child_tape_schema: str | None = None
    child_tape_digest: str | None = None
    child_tape_reference: str | None = None

    def __post_init__(self) -> None:
        if self.deployment_id < 0 or self.execution_units < 0:
            raise ValueError("external deployment counters cannot be negative")
        if not all((self.kind, self.requested_reference, self.resolved_reference, self.executor_reference)):
            raise ValueError("external deployments require complete reference identities")
        tape_fields = (
            self.child_tape_schema, self.child_tape_digest, self.child_tape_reference,
        )
        if any(tape_fields) and not all(tape_fields):
            raise ValueError("external deployment child tape identity must be complete")


@dataclass(frozen=True, slots=True)
class MachineExternalResolution:
    """A symbolic export requested by a capability-approved resolver."""

    library: str
    symbol: str


@dataclass(frozen=True, slots=True)
class MachineExternalThreadSpawn:
    """One capability-approved activation of a parked virtual core."""

    start_address: int
    parameter: int
    stack_size: int
    creation_flags: int
    thread_id: int
    handle: int

    def __post_init__(self) -> None:
        if self.start_address <= 0 or self.stack_size < 0:
            raise ValueError("thread spawn requires a target and nonnegative stack size")
        if self.thread_id <= 0 or self.handle <= 0:
            raise ValueError("thread spawn requires positive virtual identities")


@dataclass(frozen=True, slots=True)
class MachineExternalCallCompletion:
    """The complete deterministic result of one captured host interaction."""

    request_id: int
    result: int = 0
    memory_writes: tuple[MachineExternalMemoryWrite, ...] = ()
    register_writes: tuple[MachineExternalRegisterWrite, ...] = ()
    system_writes: tuple[MachineExternalStateWrite, ...] = ()
    filesystem_effects: tuple[VirtualFileEffect, ...] = ()
    registry_effects: tuple[VirtualRegistryEffect, ...] = ()
    virtual_memory_effects: tuple[VirtualMemoryEffect, ...] = ()
    environment_writes: tuple[MachineExternalEnvironmentWrite, ...] = ()
    text_writes: tuple[MachineExternalTextStateWrite, ...] = ()
    device_writes: tuple[MachineExternalDeviceWrite, ...] = ()
    control_transfer: MachineExternalControlTransfer | None = None
    deployments: tuple[MachineExternalDeployment, ...] = ()
    guest_calls: tuple[int, ...] = ()
    thread_spawns: tuple[MachineExternalThreadSpawn, ...] = ()
    resolution: MachineExternalResolution | None = None
    terminate: bool = False
    exit_code: int = 0


@dataclass(frozen=True, slots=True)
class MachineExecutionState:
    pc: int
    registers: tuple[int, ...] = (0,) * 16
    vector_registers: tuple[int, ...] = (0,) * 16
    flags: int = 0
    memory: Mapping[int, int] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    system_state: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    virtual_filesystem: VirtualFileSystemState | None = None
    virtual_registry: VirtualRegistryState | None = None
    virtual_memory: VirtualMemoryState | None = None
    environment_state: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    text_state: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    device_state: Mapping[str, bytes] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    device_generations: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    fs_base: int = 0
    gs_base: int = 0
    call_stack: tuple[int, ...] = ()
    external_requests: tuple[MachineExternalCallRequest, ...] = ()
    steps: int = 0
    termination_requested: bool = False
    halted: bool = False
    exit_code: int | None = None

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
            "fs_base": int(self.fs_base),
            "gs_base": int(self.gs_base),
            "steps": int(self.steps),
            "call_depth": len(self.call_stack),
        })
        vector_values = {
            name: value
            for index, value in enumerate(self.vector_registers)
            for name, value in (
                (f"xmm{index}_lo", int(value) & MASK64),
                (f"xmm{index}_hi", (int(value) >> 64) & MASK64),
            )
        }
        # Keep XMM halves together and before execution counters in the
        # physical register bank.
        counters = {
            key: values.pop(key) for key in ("steps", "call_depth")
        }
        values.update(vector_values)
        values.update(counters)
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
MachineColdHistoryLoader = Callable[
    [int, int], Sequence[MachineExecutionState],
]


@dataclass(frozen=True, slots=True)
class MachineTranslatedOperation:
    """One pre-bound decoded operation inside a cached basic block."""

    address: int
    instruction: Any
    execute: Callable[[MachineExecutionState], MachineExecutionResult] = field(
        repr=False, compare=False,
    )
    symbolic_effect: Any | None = None


@dataclass(frozen=True, slots=True)
class MachineTranslatedBasicBlock:
    """Immutable host translation whose every operation remains journalled."""

    entry_address: int
    operations: tuple[MachineTranslatedOperation, ...]
    code_digest: str
    cache_generation: int

    @property
    def instruction_addresses(self) -> tuple[int, ...]:
        return tuple(operation.address for operation in self.operations)


@dataclass(frozen=True, slots=True)
class MachineDispatchPlan:
    """Runtime-proven code targets discovered through a data-driven dispatch."""

    targets: tuple[int, ...]
    installed_addresses: tuple[int, ...]
    failure_reasons: tuple[str, ...] = ()


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
        translated_block_instruction_limit: int = 64,
        load_address: int | None = None,
        linked_programs: Sequence[tuple[object, int]] = (),
    ) -> None:
        if translated_block_instruction_limit <= 0:
            raise ValueError("translated block instruction limit must be positive")
        self.program = program
        self.effect_handlers = MappingProxyType(dict(effect_handlers or {}))
        self.predicate_handler = predicate_handler
        self.indirect_target_handler = indirect_target_handler
        self.external_target_resolver = external_target_resolver
        preferred_base = int(getattr(program.image, "image_base", 0))
        self.image_base = preferred_base if load_address is None else int(load_address)
        self.load_bias = self.image_base - preferred_base
        self._program_images = (
            (program, self.image_base),
            *((linked, int(base)) for linked, base in linked_programs),
        )
        instructions: dict[int, Any] = {}
        for active_program, active_base in self._program_images:
            active_bias = active_base - int(active_program.image.image_base)
            for record in active_program.functions:
                for instruction in record.report.instructions:
                    instruction = self._relocate_instruction(instruction, active_bias)
                    previous = instructions.setdefault(instruction.address, instruction)
                    if previous is not instruction and previous != instruction:
                        raise ValueError(
                            f"conflicting decoded instructions at {instruction.address:#x}"
                        )
        self.instructions = MappingProxyType(instructions)
        self._requires_mapped_code = any(
            getattr(active.image, "encoded", None) is not None
            for active, _base in self._program_images
        )
        self._dynamic_decoder = X86ReferenceDecoder()
        self._dynamic_instructions: dict[tuple[int, str], Any] = {}
        executable_pages = {
            int(instruction.address) // 4096 for instruction in instructions.values()
        }
        for active_program, image_base in self._program_images:
            image = getattr(active_program, "image", None)
            for section in getattr(image, "sections", ()):
                if not bool(getattr(section, "executable", False)):
                    continue
                begin = image_base + int(section.virtual_address)
                size = max(
                    int(getattr(section, "virtual_size", 0)),
                    int(getattr(section, "raw_size", 0)),
                )
                if size:
                    executable_pages.update(range(
                        begin // 4096, (begin + size - 1) // 4096 + 1,
                    ))
        self._executable_pages = frozenset(executable_pages)
        self.dispatch_plans: list[MachineDispatchPlan] = []
        self.translated_block_instruction_limit = int(translated_block_instruction_limit)
        self._translated_blocks: dict[int, MachineTranslatedBasicBlock] = {}
        self._translation_generation = 0
        self._translation_hits = 0
        self._translation_misses = 0

    @staticmethod
    def _relocate_instruction(instruction: Any, load_bias: int) -> Any:
        """Move decoded address identities into the runtime image namespace."""

        if not load_bias:
            return instruction
        operands = tuple(
            replace(operand, target_address=int(operand.target_address) + load_bias)
            if isinstance(operand, RelativeAddressOperand) else operand
            for operand in instruction.operands
        )
        try:
            return replace(
                instruction,
                address=int(instruction.address) + load_bias,
                operands=operands,
            )
        except TypeError:
            # Lightweight synthetic instruction records used by embedders may
            # not be dataclasses, but retain the same public field contract.
            from types import SimpleNamespace
            return SimpleNamespace(
                **{
                    **vars(instruction),
                    "address": int(instruction.address) + load_bias,
                    "operands": operands,
                }
            )

    @property
    def translation_cache_stats(self) -> Mapping[str, int]:
        return MappingProxyType({
            "generation": self._translation_generation,
            "blocks": len(self._translated_blocks),
            "hits": self._translation_hits,
            "misses": self._translation_misses,
        })

    @staticmethod
    def _ends_translated_block(semantic: MachineSemanticToken) -> bool:
        return semantic in {
            MachineSemanticToken.RETURN,
            MachineSemanticToken.DIRECT_RELATIVE_JUMP,
            MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
            MachineSemanticToken.DIRECT_RELATIVE_CALL,
            MachineSemanticToken.INDIRECT_CALL,
            MachineSemanticToken.INDIRECT_JUMP,
            MachineSemanticToken.BREAKPOINT_TRAP,
            MachineSemanticToken.SOFTWARE_INTERRUPT,
        }

    def _translate_operation(self, instruction: Any) -> MachineTranslatedOperation:
        """Pre-bind the hot non-control dispatch performed by ``step``."""

        from .machine_symbolic_effects import translated_symbolic_effect

        semantic = instruction.semantic
        if self._ends_translated_block(semantic):
            execute = lambda state: self._execute_decoded(state, instruction)
        else:
            handler = self.effect_handlers.get(int(semantic))
            if handler is None:
                def execute(state):
                    return self._reconcile_execution_result(state, MachineExecutionResult(
                        MachineExecutionStatus.BLOCKED_EFFECT,
                        state,
                        f"no emulation handler for semantic token "
                        f"{int(semantic)} ({semantic.name})",
                        int(semantic), instruction,
                    ))
            else:
                def execute(state):
                    advanced = replace(
                        state,
                        pc=instruction.address + len(instruction.encoded),
                        steps=state.steps + 1,
                    )
                    try:
                        handled = handler(advanced, instruction)
                    except (ArithmeticError, KeyError, ValueError) as error:
                        return self._reconcile_execution_result(state, MachineExecutionResult(
                            MachineExecutionStatus.TRAPPED,
                            state,
                            f"machine effect trapped at {instruction.address:#x}: {error}",
                            int(semantic), instruction,
                        ))
                    return self._reconcile_execution_result(state, MachineExecutionResult(
                        MachineExecutionStatus.RUNNING, handled,
                        instruction=instruction,
                    ))
        return MachineTranslatedOperation(
            int(instruction.address), instruction, execute,
            translated_symbolic_effect(instruction),
        )

    def _instruction_bytes_match(
        self, state: MachineExecutionState, instruction: Any,
    ) -> bool:
        try:
            observed = bytes(
                state.memory[int(instruction.address) + index]
                for index in range(len(instruction.encoded))
            )
        except KeyError:
            # Synthetic unit programs may not install a guest memory image.
            return not self._requires_mapped_code
        return observed == bytes(instruction.encoded)

    def _decode_instruction_from_state(
        self, state: MachineExecutionState, address: int,
    ) -> Any | None:
        target = int(address)
        static = self.instructions.get(target)
        if static is not None and self._instruction_bytes_match(state, static):
            return static
        if (
            target // 4096 not in self._executable_pages
            and not (
                state.virtual_memory is not None
                and state.virtual_memory.is_executable(target)
            )
        ):
            return None
        raw = bytearray()
        for index in range(15):
            try:
                raw.append(int(state.memory[target + index]))
            except KeyError:
                break
        if not raw:
            return None
        digest = sha256(bytes(raw)).hexdigest()
        cached = self._dynamic_instructions.get((target, digest))
        if cached is not None:
            return cached
        instruction, _end = self._dynamic_decoder.decode_one(
            memoryview(raw), 0, base_address=target,
        )
        if len(self._dynamic_instructions) >= 4096:
            self._dynamic_instructions.pop(next(iter(self._dynamic_instructions)))
        self._dynamic_instructions[(target, digest)] = instruction
        return instruction

    def _block_matches_state(
        self, block: MachineTranslatedBasicBlock, state: MachineExecutionState,
    ) -> bool:
        return all(
            self._instruction_bytes_match(state, operation.instruction)
            for operation in block.operations
        )

    def _reconcile_execution_result(
        self,
        source: MachineExecutionState,
        result: MachineExecutionResult,
    ) -> MachineExecutionResult:
        dynamic_executable_pages: set[int] = set()
        for active_state in (source, result.state):
            if active_state.virtual_memory is None:
                continue
            for region in active_state.virtual_memory.regions.values():
                if region.executable:
                    dynamic_executable_pages.update(range(
                        region.base // 4096, (region.end + 4095) // 4096,
                    ))
        executable_pages = self._executable_pages | dynamic_executable_pages
        touched = {
            page for page in _memory_changed_pages(
                source.memory, result.state.memory,
            )
            if page in executable_pages
        }
        if not touched:
            return result
        system_state = dict(result.state.system_state)
        for page in sorted(touched):
            key = f"machine.code_page.{page:#x}.version"
            system_state[key] = int(source.system_state.get(key, 0)) + 1
        self._translated_blocks.clear()
        self._translation_generation += 1
        return replace(
            result,
            state=replace(result.state, system_state=MappingProxyType(system_state)),
        )

    def reconcile_external_state(
        self,
        source: MachineExecutionState,
        target: MachineExecutionState,
    ) -> MachineExecutionState:
        """Version executable pages changed by a capability completion."""

        return self._reconcile_execution_result(
            source, MachineExecutionResult(MachineExecutionStatus.RUNNING, target),
        ).state

    def _execute_decoded(
        self, state: MachineExecutionState, instruction: Any,
    ) -> MachineExecutionResult:
        return self._reconcile_execution_result(
            state, self._step_decoded(state, instruction),
        )

    def execute_decoded(
        self, state: MachineExecutionState, instruction: Any,
    ) -> MachineExecutionResult:
        """Apply one already-framed instruction to the live machine state.

        This is the public insertion boundary for instruction-stream heads.
        The caller owns framing and exact decoding; this executor owns all
        architectural effects and executable-page reconciliation.  It is not
        a native-call escape hatch and it does not bypass machine semantics.
        """

        if int(instruction.address) != int(state.pc):
            raise ValueError(
                "decoded instruction address does not match the live program counter"
            )
        return self._execute_decoded(state, instruction)

    def translated_block(
        self,
        entry_address: int,
        state: MachineExecutionState | None = None,
    ) -> MachineTranslatedBasicBlock:
        """Compile and cache one decoded straight-line region by guest RIP."""

        entry = int(entry_address)
        cached = self._translated_blocks.get(entry)
        if cached is not None and (
            state is None or self._block_matches_state(cached, state)
        ):
            self._translation_hits += 1
            return cached
        if cached is not None:
            self._translated_blocks.pop(entry, None)
            self._translation_generation += 1
        instruction = (
            self.instructions.get(entry)
            if state is None else self._decode_instruction_from_state(state, entry)
        )
        if instruction is None:
            raise KeyError(f"no decoded instruction at block entry {entry:#x}")
        operations: list[MachineTranslatedOperation] = []
        digest = sha256()
        for _ in range(self.translated_block_instruction_limit):
            address = int(instruction.address)
            digest.update(address.to_bytes(8, "little", signed=False))
            digest.update(int(instruction.semantic).to_bytes(4, "little", signed=False))
            digest.update(len(instruction.encoded).to_bytes(4, "little"))
            digest.update(bytes(instruction.encoded))
            operations.append(self._translate_operation(instruction))
            if self._ends_translated_block(instruction.semantic):
                break
            successor = address + len(instruction.encoded)
            instruction = (
                self.instructions.get(successor)
                if state is None else self._decode_instruction_from_state(state, successor)
            )
            if instruction is None:
                break
        block = MachineTranslatedBasicBlock(
            entry, tuple(operations), digest.hexdigest(), self._translation_generation,
        )
        self._translated_blocks[entry] = block
        self._translation_misses += 1
        return block

    def execute_translated_block(
        self,
        state: MachineExecutionState,
        *,
        maximum_instructions: int | None = None,
    ) -> tuple[MachineExecutionResult, ...]:
        """Run a cached block but return every architectural transition."""

        limit = (
            self.translated_block_instruction_limit
            if maximum_instructions is None else int(maximum_instructions)
        )
        if limit <= 0:
            raise ValueError("translated block execution limit must be positive")
        if state.halted:
            return (self.step(state),)
        try:
            block = self.translated_block(state.pc, state)
        except (KeyError, VocabularyDecodeError):
            return (self.step(state),)
        active = state
        results: list[MachineExecutionResult] = []
        generation = self._translation_generation
        for operation in block.operations[:limit]:
            if active.pc != operation.address:
                break
            result = operation.execute(active)
            results.append(result)
            if result.status is not MachineExecutionStatus.RUNNING:
                break
            active = result.state
            # The block was validated against guest memory as a unit before
            # execution.  Reconciliation increments the generation for any
            # write to executable memory; stop before dispatching a possibly
            # modified successor rather than rereading every instruction's
            # bytes on the normal path.
            if self._translation_generation != generation:
                break
        return tuple(results) or (self.step(state),)

    def recompile_block_wasm(
        self,
        entry_address: int,
        state: MachineExecutionState | None = None,
        *,
        strict: bool = False,
        maximum_instructions: int | None = None,
    ):
        """Emit an executable, instruction-journalled Wasm block artifact."""

        from .machine_block_recompiler import lower_machine_block_to_wasm

        block = self.translated_block(entry_address, state)
        indirect_target = None
        indirect_external = False
        if state is not None and block.operations and block.operations[0].instruction.semantic in {
            MachineSemanticToken.INDIRECT_CALL,
            MachineSemanticToken.INDIRECT_JUMP,
        }:
            if self.indirect_target_handler is not None:
                try:
                    indirect_target = int(self.indirect_target_handler(
                        state, block.operations[0].instruction,
                    ))
                except (ArithmeticError, KeyError, ValueError):
                    indirect_target = None
            indirect_external = bool(
                indirect_target is not None
                and self.external_target_resolver is not None
                and self.external_target_resolver(indirect_target) is not None
            )
        executable_pages = set(self._executable_pages)
        if state is not None and state.virtual_memory is not None:
            for region in state.virtual_memory.regions.values():
                if region.executable:
                    executable_pages.update(range(
                        region.base // 4096, (region.end + 4095) // 4096,
                    ))
        return lower_machine_block_to_wasm(
            block, strict=strict,
            executable_pages=frozenset(executable_pages),
            specialization_state=state,
            maximum_instructions=maximum_instructions,
            resolved_indirect_target=indirect_target,
            indirect_external=indirect_external,
        )

    def install_dispatch_targets(self, targets: Sequence[int]) -> MachineDispatchPlan:
        """Decode validated executable targets surfaced by a runtime plan."""

        from .machine_program_graph import decode_reachable_region
        from .machine_reference_vocabulary import X86ReferenceDecoder

        merged = dict(self.instructions)
        installed: set[int] = set()
        failures: list[str] = []
        normalized = tuple(dict.fromkeys(int(target) for target in targets))
        for target in normalized:
            if target in merged:
                continue
            candidates = tuple(
                (active_program.image, active_base, target - active_base)
                for active_program, active_base in self._program_images
                if (
                    (section := active_program.image.section_for_rva(target - active_base))
                    is not None and section.executable
                )
            )
            if len(candidates) != 1:
                failures.append(
                    f"dispatch target {target:#x} maps to {len(candidates)} executable PE images"
                )
                continue
            image, image_base, rva = candidates[0]
            section = image.section_for_rva(rva)
            assert section is not None and section.executable
            owner = image.runtime_function_for_rva(rva)
            if owner is not None:
                begin_rva, end_rva = owner.begin_rva, owner.end_rva
            else:
                begin_rva = section.virtual_address
                end_rva = section.virtual_address + section.raw_size
            file_offset = image.file_offset_for_rva(begin_rva)
            if file_offset is None:
                failures.append(f"dispatch target {target:#x} has no file-backed code")
                continue
            report = decode_reachable_region(
                X86ReferenceDecoder(),
                image.encoded[file_offset:file_offset + end_rva - begin_rva],
                base_address=image_base + begin_rva,
                entry_offsets=(rva - begin_rva,),
            )
            failures.extend(
                f"{failure.address:#x}: {failure.reason}"
                for failure in report.failures
            )
            for instruction in report.instructions:
                previous = merged.setdefault(instruction.address, instruction)
                if previous != instruction:
                    raise ValueError(
                        f"dispatch decoding conflicts at {instruction.address:#x}"
                    )
                installed.add(instruction.address)
        self.instructions = MappingProxyType(merged)
        if installed:
            self._translated_blocks.clear()
            self._translation_generation += 1
        plan = MachineDispatchPlan(
            normalized, tuple(sorted(installed)), tuple(failures),
        )
        self.dispatch_plans.append(plan)
        return plan

    def initial_state(self) -> MachineExecutionState:
        return MachineExecutionState(
            pc=self.image_base + self.program.image.entrypoint_rva,
        )

    @staticmethod
    def _relative_target(instruction: Any) -> int | None:
        return next((
            operand.target_address
            for operand in instruction.operands
            if isinstance(operand, RelativeAddressOperand)
        ), None)

    @staticmethod
    def _stack_arguments(state: MachineExecutionState, count: int = 8) -> tuple[int, ...]:
        values: list[int] = []
        # At an external callee entry: return address, 32-byte home space,
        # then argument five. Stop at the first unmapped word.
        begin = state.registers[4] + 0x28
        for argument in range(count):
            address = begin + argument * 8
            try:
                value = int.from_bytes(
                    bytes(state.memory[address + index] for index in range(8)),
                    "little",
                )
            except KeyError:
                break
            values.append(value)
        return tuple(values)

    def step(self, state: MachineExecutionState) -> MachineExecutionResult:
        if state.halted:
            return MachineExecutionResult(
                MachineExecutionStatus.HALTED, state,
                f"guest process exited with code {int(state.exit_code or 0)}",
            )
        try:
            instruction = self._decode_instruction_from_state(state, state.pc)
        except VocabularyDecodeError as error:
            return MachineExecutionResult(
                MachineExecutionStatus.BLOCKED_CONTROL,
                state,
                f"guest executable bytes at {state.pc:#x} do not decode: {error}",
            )
        if instruction is None:
            return MachineExecutionResult(
                MachineExecutionStatus.BLOCKED_CONTROL,
                state,
                f"no decoded instruction at program counter {state.pc:#x}",
            )
        return self._execute_decoded(state, instruction)

    def _step_decoded(
        self, state: MachineExecutionState, instruction: Any,
    ) -> MachineExecutionResult:
        next_pc = instruction.address + len(instruction.encoded)
        semantic = instruction.semantic
        advanced = replace(state, pc=next_pc, steps=state.steps + 1)

        if semantic is MachineSemanticToken.RETURN:
            handler = self.effect_handlers.get(int(semantic))
            handled = handler(advanced, instruction) if handler is not None else advanced
            if not state.call_stack:
                if int(state.system_state.get("windows.thread.auxiliary", 0)):
                    system_state = dict(handled.system_state)
                    exit_code = int(handled.registers[0] & 0xFFFFFFFF)
                    call_count = int(system_state.get(
                        "windows.loader.startup_call_count", 0,
                    ))
                    if call_count and not int(system_state.get(
                        "windows.thread.detach_started", 0,
                    )):
                        # A clean Windows thread exit notifies every loaded
                        # image in the exiting thread before its handle becomes
                        # signalled. Reuse the validated loader call catalog in
                        # reverse order and preserve RAX before callbacks alter it.
                        # Reverse module initialization order while preserving
                        # the PE-mandated order of callbacks inside each TLS
                        # callback array (and their per-module DllMain edge).
                        groups: list[list[int]] = []
                        for candidate in range(call_count):
                            candidate_prefix = f"windows.loader.startup_call.{candidate}"
                            module_base = int(system_state[
                                f"{candidate_prefix}.module_base"
                            ])
                            if not groups or int(system_state[
                                "windows.loader.startup_call."
                                f"{groups[-1][0]}.module_base"
                            ]) != module_base:
                                groups.append([])
                            groups[-1].append(candidate)
                        detach_calls = tuple(
                            candidate
                            for group in reversed(groups)
                            for candidate in group
                        )
                        call_index = detach_calls[0]
                        prefix = f"windows.loader.startup_call.{call_index}"
                        registers = list(handled.registers)
                        registers[1] = int(system_state[f"{prefix}.module_base"])
                        registers[2] = 3  # DLL_THREAD_DETACH
                        registers[8] = 0
                        registers[4] = (registers[4] - 8) & MASK64
                        memory = handled.memory.write_unsigned(
                            registers[4], 64, MACHINE_LOADER_CALLBACK_RETURN,
                        )
                        system_state.update({
                            "windows.thread.detach_started": 1,
                            "windows.thread.detach_complete": 0,
                            "windows.thread.pending_exit_code": exit_code,
                            "windows.loader.startup_reason": 3,
                            "windows.loader.startup_call_index": call_index,
                            "windows.loader.completion_action": 1,
                            "windows.loader.detach_call_count": len(detach_calls),
                            "windows.loader.detach_call_cursor": 0,
                            "windows.loader.tls_callback_index": 0,
                            "windows.loader.tls_callbacks_complete": 0,
                            "windows.loader.startup_calls_complete": 0,
                        })
                        for detach_index, startup_index in enumerate(detach_calls):
                            system_state[
                                f"windows.loader.detach_call.{detach_index}.startup_index"
                            ] = startup_index
                        return MachineExecutionResult(
                            MachineExecutionStatus.RUNNING,
                            replace(
                                handled,
                                pc=int(system_state[f"{prefix}.address"]),
                                registers=tuple(registers),
                                memory=memory,
                                call_stack=(MACHINE_LOADER_CALLBACK_RETURN,),
                                system_state=MappingProxyType(system_state),
                            ),
                            "auxiliary guest thread entered detach callbacks",
                            instruction=instruction,
                        )
                    exit_code = int(system_state.get(
                        "windows.thread.pending_exit_code", exit_code,
                    ))
                    system_state["windows.thread.active"] = 0
                    system_state["windows.thread.exit_code"] = exit_code
                    system_state["windows.thread.detach_complete"] = 1
                    thread_id = int(system_state.get("windows.thread.id", 0))
                    if thread_id:
                        system_state[f"windows.thread.{thread_id}.active"] = 0
                        system_state[f"windows.thread.{thread_id}.exit_code"] = exit_code
                    return MachineExecutionResult(
                        MachineExecutionStatus.RUNNING,
                        replace(
                            handled,
                            halted=True,
                            exit_code=exit_code,
                            system_state=MappingProxyType(system_state),
                        ),
                        "auxiliary guest thread returned and parked",
                        instruction=instruction,
                    )
                return MachineExecutionResult(
                    MachineExecutionStatus.HALTED, advanced,
                    "returned from the outermost emulated frame", instruction=instruction,
                )
            if state.call_stack[-1] == MACHINE_LOADER_CALLBACK_RETURN:
                system_state = dict(handled.system_state)
                call_index = int(system_state.get(
                    "windows.loader.startup_call_index", 0,
                ))
                current_prefix = f"windows.loader.startup_call.{call_index}"
                if (
                    int(system_state.get(f"{current_prefix}.requires_success", 0))
                    and int(system_state.get("windows.loader.startup_reason", 1)) == 1
                    and int(handled.registers[0]) == 0
                ):
                    system_state["windows.loader.startup_failure_index"] = call_index
                    system_state["windows.loader.startup_failure_kind"] = int(
                        system_state.get(f"{current_prefix}.kind", 0)
                    )
                    return MachineExecutionResult(
                        MachineExecutionStatus.TRAPPED,
                        replace(
                            handled,
                            system_state=MappingProxyType(system_state),
                        ),
                        "DLL process-attach entry point returned false",
                        instruction=instruction,
                    )
                if int(system_state.get(f"{current_prefix}.kind", 0)) == 1:
                    system_state["windows.loader.tls_callback_index"] = int(
                        system_state.get("windows.loader.tls_callback_index", 0)
                    ) + 1
                call_count = int(system_state.get(
                    "windows.loader.startup_call_count", 0,
                ))
                completion_action = int(system_state.get(
                    "windows.loader.completion_action", 0,
                ))
                if completion_action == 1:
                    detach_cursor = int(system_state.get(
                        "windows.loader.detach_call_cursor", 0,
                    )) + 1
                    system_state["windows.loader.detach_call_cursor"] = detach_cursor
                    detach_count = int(system_state.get(
                        "windows.loader.detach_call_count", 0,
                    ))
                    call_index = (
                        int(system_state[
                            f"windows.loader.detach_call.{detach_cursor}.startup_index"
                        ])
                        if detach_cursor < detach_count else -1
                    )
                else:
                    call_index += 1
                system_state["windows.loader.startup_call_index"] = call_index
                registers = list(handled.registers)
                if 0 <= call_index < call_count:
                    prefix = f"windows.loader.startup_call.{call_index}"
                    callback = int(system_state[f"{prefix}.address"])
                    module_base = int(system_state[f"{prefix}.module_base"])
                    registers[1] = module_base
                    registers[2] = int(system_state.get(
                        "windows.loader.startup_reason", 1,
                    ))
                    registers[8] = 0
                    registers[4] = (registers[4] - 8) & MASK64
                    target = callback
                    stack = state.call_stack
                else:
                    if completion_action == 1:
                        exit_code = int(system_state.get(
                            "windows.thread.pending_exit_code", 0,
                        ))
                        system_state.update({
                            "windows.thread.active": 0,
                            "windows.thread.exit_code": exit_code,
                            "windows.thread.detach_complete": 1,
                            "windows.loader.tls_callbacks_complete": 1,
                            "windows.loader.startup_calls_complete": 1,
                            "windows.loader.completion_action": 0,
                        })
                        thread_id = int(system_state.get("windows.thread.id", 0))
                        if thread_id:
                            system_state[f"windows.thread.{thread_id}.active"] = 0
                            system_state[f"windows.thread.{thread_id}.exit_code"] = exit_code
                        return MachineExecutionResult(
                            MachineExecutionStatus.RUNNING,
                            replace(
                                handled,
                                call_stack=state.call_stack[:-1],
                                system_state=MappingProxyType(system_state),
                                halted=True,
                                exit_code=exit_code,
                            ),
                            "auxiliary guest thread detached and parked",
                            instruction=instruction,
                        )
                    target = int(system_state["windows.loader.entrypoint"])
                    stack = state.call_stack[:-1]
                    if int(system_state.get("windows.thread.auxiliary", 0)):
                        registers[1] = int(system_state.get(
                            "windows.thread.start_parameter", 0,
                        ))
                    system_state["windows.loader.tls_callbacks_complete"] = 1
                    system_state["windows.loader.startup_calls_complete"] = 1
                return MachineExecutionResult(
                    MachineExecutionStatus.RUNNING,
                    replace(
                        handled,
                        pc=target,
                        registers=tuple(registers),
                        call_stack=stack,
                        system_state=MappingProxyType(system_state),
                    ),
                    instruction=instruction,
                )
            if (
                state.call_stack[-1] == MACHINE_TERMINATION_RETURN
                and state.termination_requested
            ):
                return MachineExecutionResult(
                    MachineExecutionStatus.HALTED,
                    replace(
                        handled,
                        call_stack=state.call_stack[:-1],
                        termination_requested=False,
                        halted=True,
                    ),
                    f"guest process exited with code {int(state.exit_code or 0)}",
                    instruction=instruction,
                )
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
            try:
                target_instruction = (
                    None if target is None
                    else self._decode_instruction_from_state(state, target)
                )
            except VocabularyDecodeError:
                target_instruction = None
            if target is None or target_instruction is None:
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
                    stack_arguments=self._stack_arguments(handled),
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
        try:
            handled = handler(advanced, instruction)
        except (ArithmeticError, KeyError, ValueError) as error:
            return MachineExecutionResult(
                MachineExecutionStatus.TRAPPED,
                state,
                f"machine effect trapped at {instruction.address:#x}: {error}",
                int(semantic),
                instruction,
            )
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


@dataclass(frozen=True, slots=True)
class MachineSharedMemoryWrite:
    """One changed byte range in a deterministic virtual-core turn."""

    core_index: int
    address: int
    byte_length: int
    before_digest: str
    after_digest: str


@dataclass(frozen=True, slots=True)
class MachineSharedMemoryConflict:
    """Overlapping writes ordered by the explicit core schedule."""

    address: int
    byte_length: int
    earlier_core: int
    later_core: int


@dataclass(frozen=True, slots=True)
class MachineSharedMemoryCommit:
    """Auditable sequentially-consistent memory barrier for guest threads."""

    cycle_index: int
    core_order: tuple[int, ...]
    core_positions: tuple[int, ...]
    writes: tuple[MachineSharedMemoryWrite, ...] = ()
    conflicts: tuple[MachineSharedMemoryConflict, ...] = ()

    def to_mapping(self) -> Mapping[str, object]:
        return MappingProxyType({
            "cycle_index": self.cycle_index,
            "core_order": self.core_order,
            "core_positions": self.core_positions,
            "writes": tuple({
                "core": item.core_index,
                "address": item.address,
                "byte_length": item.byte_length,
                "before_digest": item.before_digest,
                "after_digest": item.after_digest,
            } for item in self.writes),
            "conflicts": tuple({
                "address": item.address,
                "byte_length": item.byte_length,
                "earlier_core": item.earlier_core,
                "later_core": item.later_core,
            } for item in self.conflicts),
        })


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
    _history_base: int = 0
    _history_tip: int | None = None
    maximum_hot_states: int | None = None
    cold_history_loader: MachineColdHistoryLoader | None = field(
        default=None, repr=False,
    )

    def __post_init__(self) -> None:
        resident_tip = self._history_base + len(self._states)
        if self._history_tip is None:
            self._history_tip = resident_tip
        elif self._history_tip < resident_tip:
            raise ValueError("machine history tip cannot precede resident states")

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
        return self._history_base + self._position

    @property
    def history_length(self) -> int:
        assert self._history_tip is not None
        return self._history_tip

    @property
    def hot_history_range(self) -> tuple[int, int]:
        """Half-open absolute position range currently resident in memory."""

        return self._history_base, self._history_base + len(self._states)

    @property
    def edges(self) -> tuple[MachineExecutionEdge, ...]:
        return tuple(self._edges)

    @property
    def latest_edge(self) -> MachineExecutionEdge | None:
        return self._edges[self._position - 1] if self._position > 0 else None

    def configure_hot_history(
        self,
        maximum_states: int | None,
        *,
        cold_history_loader: MachineColdHistoryLoader | None = None,
    ) -> None:
        if maximum_states is not None and int(maximum_states) < 2:
            raise ValueError("hot machine history must retain at least two states")
        self.maximum_hot_states = (
            None if maximum_states is None else int(maximum_states)
        )
        self.cold_history_loader = cold_history_loader
        self._prune_hot_history()

    def restore_cold_tip(
        self,
        state: MachineExecutionState,
        *,
        position: int,
    ) -> MachineExecutionState:
        """Install one persisted tip while retaining its absolute history axis."""

        absolute = int(position)
        if absolute < 0:
            raise ValueError("persisted machine history position cannot be negative")
        self._states[:] = [state]
        self._edges[:] = []
        self._position = 0
        self._history_base = absolute
        self._history_tip = absolute + 1
        return state

    @staticmethod
    def _cold_edge_status(target: MachineExecutionState) -> MachineExecutionStatus:
        if target.halted:
            return MachineExecutionStatus.HALTED
        if target.external_requests:
            return MachineExecutionStatus.WAITING_EXTERNAL
        return MachineExecutionStatus.RUNNING

    def _hydrate_cold_history(self, requested_start: int) -> None:
        if requested_start >= self._history_base:
            return
        if self.cold_history_loader is None:
            raise IndexError(
                "machine history position is cold and no tape loader is configured"
            )
        loaded = tuple(self.cold_history_loader(
            max(0, int(requested_start)), self._history_base,
        ))
        if not loaded:
            raise IndexError("machine tape did not provide the requested cold history")
        # The exact tape remains authority for the evicted future. Keeping it
        # resident while repeatedly prepending would defeat the memory bound.
        current = self.state
        self._states[:] = [current]
        self._edges[:] = []
        self._position = 0
        start = self._history_base - len(loaded)
        if start < requested_start:
            raise ValueError("machine tape loader returned too many cold states")
        combined = (*loaded, *self._states)
        prefix_edges = tuple(
            MachineExecutionEdge(
                source, target, self._cold_edge_status(target),
                self.executor.instructions.get(source.pc),
            )
            for source, target in zip(combined, combined[1:len(loaded) + 1])
        )
        self._states[:0] = loaded
        self._edges[:0] = prefix_edges
        self._position += len(loaded)
        self._history_base = start

    def _truncate_future(self) -> None:
        """Turn execution after rewind into a new branch, including cold future."""

        del self._states[self._position + 1:]
        del self._edges[self._position:]
        self._history_tip = self.position + 1

    def _prune_hot_history(self) -> None:
        maximum = self.maximum_hot_states
        if maximum is None or len(self._states) <= maximum:
            return
        # Never discard a future while positioned inside it. Branch truncation
        # happens first; pruning is safe only at the newest retained state.
        if self._position != len(self._states) - 1:
            return
        count = len(self._states) - maximum
        del self._states[:count]
        del self._edges[:count]
        self._position -= count
        self._history_base += count

    def commit_execution_result(
        self, result: MachineExecutionResult,
    ) -> MachineExecutionResult:
        """Journal an already scheduled instruction result exactly once."""

        source = self.state
        self._truncate_future()
        self._states.append(result.state)
        self._edges.append(MachineExecutionEdge(
            source, result.state, result.status, result.instruction,
        ))
        self._position += 1
        self._history_tip = self.position + 1
        self._prune_hot_history()
        return result

    def step_forward(self) -> MachineExecutionResult:
        return self.commit_execution_result(self.executor.step(self.state))

    def step_block_forward(
        self,
        maximum_instructions: int,
        *,
        transition_observer: Callable[[], None] | None = None,
    ) -> tuple[MachineExecutionResult, ...]:
        """Execute a cached block while retaining one reversible edge per instruction."""

        source = self.state
        results = self.executor.execute_translated_block(
            source, maximum_instructions=maximum_instructions,
        )
        self._truncate_future()
        for result in results:
            self._states.append(result.state)
            self._edges.append(MachineExecutionEdge(
                source, result.state, result.status, result.instruction,
            ))
            self._position += 1
            self._history_tip = self.position + 1
            self._prune_hot_history()
            source = result.state
            if transition_observer is not None:
                transition_observer()
        return results

    def step_backward(self) -> MachineExecutionState:
        if self._position == 0 and self._history_base > 0:
            maximum = self.maximum_hot_states or 2
            self._hydrate_cold_history(max(0, self._history_base - maximum + 1))
        if self._position == 0:
            raise IndexError("machine executor is already at its initial state")
        self._position -= 1
        return self.state

    def commit_external_completion(self, state: MachineExecutionState) -> MachineExecutionState:
        """Journal a shell-supplied completion as a reversible graph edge."""

        source = self.state
        self._truncate_future()
        self._states.append(state)
        self._edges.append(MachineExecutionEdge(
            source, state, MachineExecutionStatus.RUNNING, None,
        ))
        self._position += 1
        self._history_tip = self.position + 1
        self._prune_hot_history()
        return state

    def commit_shell_effect(self, state: MachineExecutionState) -> MachineExecutionState:
        """Journal shell-owned device input without completing a guest call."""

        source = self.state
        self._truncate_future()
        self._states.append(state)
        self._edges.append(MachineExecutionEdge(
            source, state, MachineExecutionStatus.RUNNING, None,
        ))
        self._position += 1
        self._history_tip = self.position + 1
        self._prune_hot_history()
        return state

    def commit_recompiled_journal(
        self,
        artifact,
        encoded: bytes,
        *,
        transition_observer: Callable[[], None] | None = None,
    ) -> tuple[MachineExecutionResult, ...]:
        """Validate and journal instruction checkpoints emitted by a backend.

        Recompiled execution is never committed as one opaque block edge. The
        artifact reconstructs and authenticates each architectural state, and
        this bridge binds it back to the decoded guest instruction before
        appending the normal reversible edge.
        """

        targets = artifact.states_from_journal(bytes(encoded), self.state)
        if len(targets) != len(artifact.witnesses):
            raise ValueError("recompiled machine journal/witness count mismatch")
        results = []
        for witness, target in zip(artifact.witnesses, targets):
            if self.state.pc != int(witness.address):
                raise ValueError("recompiled machine journal is not contiguous with history")
            try:
                instruction = self.executor._decode_instruction_from_state(
                    self.state, int(witness.address),
                )
            except VocabularyDecodeError:
                instruction = None
            if instruction is None:
                raise ValueError(
                    "recompiled machine journal names unknown guest code at "
                    f"{int(witness.address):#x}"
                )
            if bytes(instruction.encoded) != bytes(witness.encoded):
                raise ValueError("recompiled machine witness disagrees with decoded bytes")
            if not self.executor._instruction_bytes_match(self.state, instruction):
                raise ValueError("recompiled machine witness no longer matches guest memory")
            target = replace(target, system_state=self.state.system_state)
            target = self.executor.reconcile_external_state(self.state, target)
            result = MachineExecutionResult(
                MachineExecutionStatus.RUNNING, target,
                "recompiled instruction checkpoint", instruction=instruction,
            )
            results.append(self.commit_execution_result(result))
            if transition_observer is not None:
                transition_observer()
        return tuple(results)

    def seek_history(self, position: int) -> MachineExecutionState:
        absolute = int(position)
        assert self._history_tip is not None
        if not 0 <= absolute < self._history_tip:
            raise IndexError("machine execution history position is out of range")
        resident_end = self._history_base + len(self._states)
        if not self._history_base <= absolute < resident_end:
            if self.cold_history_loader is None:
                raise IndexError(
                    "machine history position is cold and no tape loader is configured"
                )
            loaded = tuple(self.cold_history_loader(absolute, absolute + 1))
            if len(loaded) != 1:
                raise IndexError("machine tape did not provide the requested history state")
            self._states[:] = [loaded[0]]
            self._edges[:] = []
            self._history_base = absolute
            self._position = 0
            return self.state
        local = absolute - self._history_base
        self._position = local
        return self.state

    def fork(self) -> "ReversibleMachineExecutor":
        return ReversibleMachineExecutor(
            executor=self.executor,
            _states=list(self._states[:self._position + 1]),
            _edges=list(self._edges[:self._position]),
            _position=self._position,
            _history_base=self._history_base,
            _history_tip=self.position + 1,
            maximum_hot_states=self.maximum_hot_states,
            cold_history_loader=self.cold_history_loader,
        )


def _memory_equal(left: Mapping[int, int], right: Mapping[int, int]) -> bool:
    if left is right:
        return True
    left_pages, right_pages = getattr(left, "pages", None), getattr(right, "pages", None)
    if left_pages is not None and right_pages is not None:
        return (
            getattr(left, "page_size", None) == getattr(right, "page_size", None)
            and left_pages == right_pages
        )
    return left == right


def _memory_changed_pages(
    before: Mapping[int, int], after: Mapping[int, int],
) -> tuple[int, ...]:
    """Return changed 4 KiB pages, using immediate COW provenance when valid."""

    if before is after:
        return ()
    before_pages = getattr(before, "pages", None)
    after_pages = getattr(after, "pages", None)
    before_page_size = getattr(before, "page_size", None)
    after_page_size = getattr(after, "page_size", None)
    if (
        before_pages is not None
        and after_pages is not None
        and before_page_size == after_page_size == 4096
        and getattr(after, "_parent_pages_identity", 0) == id(before_pages)
    ):
        return tuple(getattr(after, "_changed_pages", ()))
    changed: set[int] = set()
    for address, old, new in _memory_changed_ranges(before, after):
        length = max(len(old or b""), len(new or b""))
        if length:
            changed.update(range(
                address // 4096, (address + length - 1) // 4096 + 1,
            ))
    return tuple(sorted(changed))


def _memory_changed_ranges(
    before: Mapping[int, int], after: Mapping[int, int],
) -> tuple[tuple[int, bytes | None, bytes | None], ...]:
    """Find compact changed ranges, exploiting copy-on-write guest pages."""

    if before is after:
        return ()
    before_pages, after_pages = getattr(before, "pages", None), getattr(after, "pages", None)
    page_size = getattr(before, "page_size", None)
    if (
        before_pages is not None and after_pages is not None
        and page_size == getattr(after, "page_size", None)
    ):
        changes: list[tuple[int, bytes | None, bytes | None]] = []
        for page in sorted(set(before_pages) | set(after_pages)):
            old = before_pages.get(page)
            new = after_pages.get(page)
            if old == new:
                continue
            address = int(page) * int(page_size)
            if old is None or new is None:
                changes.append((address, old, new))
                continue
            cursor = 0
            while cursor < page_size:
                if old[cursor] == new[cursor]:
                    cursor += 1
                    continue
                start = cursor
                while cursor < page_size and old[cursor] != new[cursor]:
                    cursor += 1
                changes.append((
                    address + start, old[start:cursor], new[start:cursor],
                ))
        return tuple(changes)

    changes = []
    active_start = None
    old_bytes = bytearray()
    new_bytes = bytearray()
    previous = None
    for address in sorted(set(before) | set(after)):
        old = before.get(address)
        new = after.get(address)
        if old == new:
            continue
        if old is None or new is None:
            if active_start is not None:
                changes.append((active_start, bytes(old_bytes), bytes(new_bytes)))
                active_start = None
                old_bytes = bytearray()
                new_bytes = bytearray()
            changes.append((
                int(address),
                None if old is None else bytes((int(old),)),
                None if new is None else bytes((int(new),)),
            ))
            previous = int(address)
            continue
        if (
            active_start is None or previous is None or address != previous + 1
        ):
            if active_start is not None:
                changes.append((active_start, bytes(old_bytes), bytes(new_bytes)))
            active_start = int(address)
            old_bytes = bytearray() if old is None else bytearray((int(old),))
            new_bytes = bytearray() if new is None else bytearray((int(new),))
        else:
            old_bytes.append(int(old))
            new_bytes.append(int(new))
        previous = int(address)
    if active_start is not None:
        changes.append((active_start, bytes(old_bytes), bytes(new_bytes)))
    return tuple(changes)


def _memory_range_digest(data: bytes | None) -> str:
    return sha256(b"<unmapped>" if data is None else b"<mapped>" + data).hexdigest()


@dataclass(slots=True)
class MachineVirtualMulticore:
    """Guest threads with deterministic memory, distinct from world forks."""

    cores: tuple[ReversibleMachineExecutor, ...]
    shared_memory: bool = True
    maximum_shared_memory_commits: int = 4096
    shared_memory_commits: list[MachineSharedMemoryCommit] = field(default_factory=list)
    _cycle_index: int = 0

    @classmethod
    def create(
        cls,
        executor: MachineExecutionOrchestrator,
        *,
        core_count: int,
        initial_states: Sequence[MachineExecutionState] | None = None,
        shared_memory: bool = True,
        maximum_shared_memory_commits: int = 4096,
    ) -> "MachineVirtualMulticore":
        if core_count <= 0:
            raise ValueError("virtual multicore requires at least one core")
        states = tuple(initial_states or ())
        if states and len(states) != core_count:
            raise ValueError("initial_states must contain one state per core")
        if maximum_shared_memory_commits <= 0:
            raise ValueError("shared-memory commit capacity must be positive")
        cores = tuple(
            ReversibleMachineExecutor.create(
                executor, states[index] if states else None,
            )
            for index in range(core_count)
        )
        if shared_memory and cores and any(
            not _memory_equal(cores[0].state.memory, core.state.memory)
            for core in cores[1:]
        ):
            raise ValueError("guest threads must begin with identical shared memory")
        return cls(
            cores, bool(shared_memory), int(maximum_shared_memory_commits),
        )

    @property
    def last_shared_memory_commit(self) -> MachineSharedMemoryCommit | None:
        return self.shared_memory_commits[-1] if self.shared_memory_commits else None

    def cycle_forward(self) -> tuple[MachineExecutionResult, ...]:
        """Advance every execution head once across the virtual core barrier."""

        if not self.shared_memory or len(self.cores) < 2:
            return tuple(core.step_forward() for core in self.cores)
        sources = tuple(core.state for core in self.cores)
        shared = sources[0].memory
        if any(not _memory_equal(shared, state.memory) for state in sources[1:]):
            raise RuntimeError("virtual-core shared memory diverged before its barrier")
        provisional: list[MachineExecutionResult] = []
        shared_system = {
            key: value for key, value in sources[0].system_state.items()
            if _shared_system_key(key)
        }
        writes: list[MachineSharedMemoryWrite] = []
        intervals: list[tuple[int, int, int]] = []
        conflicts: list[MachineSharedMemoryConflict] = []
        for core_index, (core, source) in enumerate(zip(self.cores, sources)):
            scheduled_system = dict(source.system_state)
            scheduled_system.update(shared_system)
            scheduled = replace(
                source, memory=shared,
                system_state=MappingProxyType(scheduled_system),
            )
            if (
                not int(scheduled.system_state.get("windows.thread.active", 1))
                or int(scheduled.system_state.get("windows.thread.waiting_request", 0))
            ):
                result = MachineExecutionResult(
                    MachineExecutionStatus.RUNNING,
                    scheduled,
                    "virtual thread core is parked or waiting",
                )
            else:
                result = core.executor.step(scheduled)
            for address, before, after in _memory_changed_ranges(shared, result.state.memory):
                length = max(len(before or b""), len(after or b""))
                end = address + length
                for earlier_start, earlier_end, earlier_core in intervals:
                    overlap_start = max(address, earlier_start)
                    overlap_end = min(end, earlier_end)
                    if overlap_start < overlap_end:
                        conflicts.append(MachineSharedMemoryConflict(
                            overlap_start, overlap_end - overlap_start,
                            earlier_core, core_index,
                        ))
                intervals.append((address, end, core_index))
                writes.append(MachineSharedMemoryWrite(
                    core_index, address, length,
                    _memory_range_digest(before), _memory_range_digest(after),
                ))
            shared = result.state.memory
            shared_system.update({
                key: value for key, value in result.state.system_state.items()
                if _shared_system_key(key)
            })
            provisional.append(result)
        adjusted = tuple(
            replace(result, state=replace(
                result.state, memory=shared,
                system_state=MappingProxyType({
                    **dict(result.state.system_state), **shared_system,
                }),
            ))
            for result in provisional
        )
        committed = tuple(
            core.commit_execution_result(result)
            for core, result in zip(self.cores, adjusted)
        )
        self._cycle_index += 1
        commit = MachineSharedMemoryCommit(
            self._cycle_index,
            tuple(range(len(self.cores))),
            tuple(core.position for core in self.cores),
            tuple(writes), tuple(conflicts),
        )
        self.shared_memory_commits.append(commit)
        if len(self.shared_memory_commits) > self.maximum_shared_memory_commits:
            del self.shared_memory_commits[
                :len(self.shared_memory_commits) - self.maximum_shared_memory_commits
            ]
        return committed

    def cycle_backward(self) -> tuple[MachineExecutionState, ...]:
        """Restore every execution head one cycle across the same barrier."""

        if any(core.position == 0 for core in self.cores):
            raise IndexError("at least one virtual core is already at initial state")
        positions = tuple(core.position for core in self.cores)
        states = tuple(core.step_backward() for core in self.cores)
        commit = self.last_shared_memory_commit
        if commit is not None and commit.core_positions == positions:
            self.shared_memory_commits.pop()
            self._cycle_index = max(0, self._cycle_index - 1)
        return states

    def synchronize_shared_memory(self, source_core: int) -> tuple[int, ...]:
        """Broadcast an out-of-barrier capability write as journalled edges."""

        if not 0 <= source_core < len(self.cores):
            raise IndexError("virtual core index is out of range")
        if not self.shared_memory or len(self.cores) < 2:
            return ()
        memory = self.cores[source_core].state.memory
        source_system = self.cores[source_core].state.system_state
        source_filesystem = self.cores[source_core].state.virtual_filesystem
        source_registry = self.cores[source_core].state.virtual_registry
        source_virtual_memory = self.cores[source_core].state.virtual_memory
        shared_system = {
            key: value for key, value in source_system.items()
            if _shared_system_key(key)
        }
        synchronized = []
        for index, core in enumerate(self.cores):
            if index == source_core:
                continue
            system_state = dict(core.state.system_state)
            version_changed = any(
                system_state.get(key) != value for key, value in shared_system.items()
            )
            filesystem_changed = core.state.virtual_filesystem != source_filesystem
            registry_changed = core.state.virtual_registry != source_registry
            virtual_memory_changed = core.state.virtual_memory != source_virtual_memory
            if (
                _memory_equal(core.state.memory, memory)
                and not version_changed and not filesystem_changed
                and not registry_changed and not virtual_memory_changed
            ):
                continue
            system_state.update(shared_system)
            core.commit_shell_effect(replace(
                core.state, memory=memory,
                system_state=MappingProxyType(system_state),
                virtual_filesystem=source_filesystem,
                virtual_registry=source_registry,
                virtual_memory=source_virtual_memory,
            ))
            synchronized.append(index)
        return tuple(synchronized)

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
    "MachineColdHistoryLoader",
    "MachineExecutionOrchestrator",
    "MachineExternalCallRequest",
    "MachineExternalCallCompletion",
    "MachineExternalControlTransfer",
    "MachineExternalDeployment",
    "MachineExternalMemoryWrite",
    "MachineExternalRegisterWrite",
    "MachineExternalStateWrite",
    "MachineExternalReference",
    "MachineExternalThreadSpawn",
    "MachineExternalResolution",
    "MachineExternalTargetResolver",
    "MachineExecutionEdge",
    "MachineExecutionResult",
    "MachineExecutionState",
    "MachineExecutionStatus",
    "MachineTranslatedBasicBlock",
    "MachineTranslatedOperation",
    "MACHINE_TERMINATION_RETURN",
    "MACHINE_LOADER_CALLBACK_RETURN",
    "MachineDispatchPlan",
    "MachineVirtualMulticore",
    "MachineIndirectTargetHandler",
    "MachineOrchestrationMode",
    "MachineOrchestrationResult",
    "MachinePredicateHandler",
    "MachineSharedMemoryCommit",
    "MachineSharedMemoryConflict",
    "MachineSharedMemoryWrite",
    "ReversibleMachineExecutor",
    "orchestrate_machine_program",
]
