"""Executor-side insertion of recompiled machine streams.

This is deliberately neither a PE patch nor an SSA interpreter.  A trigger at
an original guest address selects a separately laid-out instruction stream.
Every replacement instruction is framed by the repository's tensor read head,
decoded from the bytes that head consumed, and then applied by the ordinary
machine-state executor.  Registers, flags, memory, calls, and branches are
therefore the live guest state; only instruction provenance changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..common.tensors import AbstractTensor
from ..transmogrifier.ssa import IRModule
from .machine_execution import (
    MachineExecutionOrchestrator,
    MachineExecutionResult,
    MachineExecutionState,
    MachineExecutionStatus,
)
from .machine_reference_vocabulary import X86ReferenceDecoder
from .pe_recompilation import PERecompilationLedger
from .x86_tensor_read_head import (
    ReadStatus,
    X86AllocatedInstruction,
    X86EncodingFields,
    X86ReadBatch,
    X86ReversibleReadHead,
    X86TensorReadHead,
    controlled_x86_64_read_head_profile,
)


@dataclass(frozen=True, slots=True)
class RecompiledMachineStream:
    """Addressed machine bytes and their repository-SSA provenance.

    ``triggers`` maps an address in the unmodified guest image to the first
    address of replacement code.  Replacement addresses need not overlap the
    original image.  ``exit_redirects`` maps a replacement control successor
    to the original continuation.  Consequently replacement code may be much
    larger than the region it replaces and may contain an arbitrary CFG.
    """

    name: str
    instructions: Mapping[int, bytes]
    triggers: Mapping[int, int]
    exit_redirects: Mapping[int, int]
    ssa_line_ids: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    witnesses: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        instructions = {
            int(address): bytes(encoded)
            for address, encoded in self.instructions.items()
        }
        if not instructions:
            raise ValueError("a recompiled machine stream cannot be empty")
        if any(not encoded for encoded in instructions.values()):
            raise ValueError("a recompiled machine instruction cannot be empty")
        triggers = {int(source): int(target) for source, target in self.triggers.items()}
        if not triggers:
            raise ValueError("a recompiled machine stream requires a trigger")
        missing = sorted(set(triggers.values()) - set(instructions))
        if missing:
            raise ValueError(
                "stream trigger targets are not emitted instruction addresses: "
                + ", ".join(f"{item:#x}" for item in missing)
            )
        decoder = X86ReferenceDecoder()
        occupied: dict[int, int] = {}
        for address, encoded in sorted(instructions.items()):
            decoded, end = decoder.decode_one(
                memoryview(encoded), 0, base_address=address,
            )
            if end != len(encoded) or bytes(decoded.encoded) != encoded:
                raise ValueError(f"stream instruction at {address:#x} is not exact")
            for byte_address in range(address, address + len(encoded)):
                previous = occupied.setdefault(byte_address, address)
                if previous != address:
                    raise ValueError(
                        f"stream instructions at {previous:#x} and {address:#x} overlap"
                    )
        line_ids = {
            int(address): tuple(str(item) for item in values)
            for address, values in self.ssa_line_ids.items()
        }
        unknown_lines = sorted(set(line_ids) - set(instructions))
        if unknown_lines:
            raise ValueError("SSA provenance names an address absent from the stream")
        witnesses = {int(address): str(value) for address, value in self.witnesses.items()}
        object.__setattr__(self, "instructions", MappingProxyType(instructions))
        object.__setattr__(self, "triggers", MappingProxyType(triggers))
        object.__setattr__(
            self, "exit_redirects",
            MappingProxyType({
                int(source): int(target)
                for source, target in self.exit_redirects.items()
            }),
        )
        object.__setattr__(self, "ssa_line_ids", MappingProxyType(line_ids))
        object.__setattr__(self, "witnesses", MappingProxyType(witnesses))

    @classmethod
    def from_recompilation_ledger(
        cls,
        module: IRModule,
        ledger: PERecompilationLedger,
        *,
        name: str,
        trigger_address: int,
        stream_entry_address: int | None = None,
        exit_redirects: Mapping[int, int] | None = None,
        continuation_address: int | None = None,
    ) -> "RecompiledMachineStream":
        """Build a stream only from proof-complete reverse-compiler output.

        Duplicate SSA occurrences are retained by the ledger but must agree on
        their physical bytes.  No unresolved occurrence is silently omitted.
        """

        unresolved = ledger.unresolved
        if unresolved:
            details = ", ".join(
                f"#{item.occurrence}@{item.machine_address:#x}"
                for item in unresolved
            )
            raise ValueError(f"recompiled stream has unresolved occurrences: {details}")
        instructions: dict[int, bytes] = {}
        witnesses: dict[int, str] = {}
        for item in ledger.occurrences:
            assert item.encoded is not None
            address = int(item.machine_address)
            encoded = bytes(item.encoded)
            previous = instructions.setdefault(address, encoded)
            if previous != encoded:
                raise ValueError(f"conflicting recompiled bytes at {address:#x}")
            previous_witness = witnesses.setdefault(address, str(item.witness))
            if previous_witness != str(item.witness):
                raise ValueError(f"conflicting recompilation witnesses at {address:#x}")
        line_ids: dict[int, list[str]] = {}
        for function_name, function in module.functions.items():
            for block_name, block in function.blocks.items():
                for ordinal, instruction in enumerate(block.instrs):
                    address = instruction.attributes.get("machine_address")
                    if address is None or int(address) not in instructions:
                        continue
                    line_ids.setdefault(int(address), []).append(
                        f"{function_name}:{block_name}:{ordinal}"
                    )
        entry = (
            min(instructions)
            if stream_entry_address is None else int(stream_entry_address)
        )
        redirects = dict(exit_redirects or {})
        if continuation_address is not None:
            decoder = X86ReferenceDecoder()
            final_address = max(instructions)
            final, _end = decoder.decode_one(
                memoryview(instructions[final_address]), 0,
                base_address=final_address,
            )
            redirects.setdefault(
                final_address + len(final.encoded), int(continuation_address),
            )
        return cls(
            str(name), instructions, {int(trigger_address): entry}, redirects,
            {address: tuple(values) for address, values in line_ids.items()},
            witnesses,
        )


class MachineStreamRoute(IntEnum):
    """The live middle-stage decision after an instruction has been read."""

    PASS_THROUGH = 0
    SSA_RECOMPILE = 1
    EXTERNAL_SSA_REFERENCE = 2


@dataclass(frozen=True, slots=True)
class SSAWriteHeadRequest:
    """Repository SSA plus the explicit facts needed to select machine code."""

    module: IRModule
    name: str
    trigger_address: int
    stream_entry_address: int | None = None
    exit_redirects: Mapping[int, int] = field(default_factory=dict)
    proven_facts_by_address: Mapping[int, Iterable[str]] = field(default_factory=dict)
    encoding_fields_by_address: Mapping[int, X86EncodingFields] = field(default_factory=dict)
    allocated_instructions_by_address: Mapping[
        int, X86AllocatedInstruction
    ] = field(default_factory=dict)
    allow_new_selection: bool = False
    # Empty retains the historical whole-module write.  A non-empty set is
    # the repository-SSA incremental boundary and is passed directly to the
    # authoritative reverse-selection ledger; it is not a second compiler.
    selected_blocks: tuple[tuple[str, str], ...] = ()


class BidirectionalSSAWriteHead:
    """Proof-gated repository-SSA to machine-stream stage."""

    def __init__(self, *, maximum_cached_fragments: int = 4096) -> None:
        if int(maximum_cached_fragments) <= 0:
            raise ValueError("write-head fragment cache bound must be positive")
        self.maximum_cached_fragments = int(maximum_cached_fragments)
        self._fragments: dict[str, RecompiledMachineStream] = {}
        self._hits = 0
        self._misses = 0

    @property
    def cache_statistics(self) -> Mapping[str, int]:
        return MappingProxyType({
            "fragments": len(self._fragments),
            "hits": self._hits,
            "misses": self._misses,
        })

    @staticmethod
    def _request_identity(request: SSAWriteHeadRequest) -> str:
        from .machine_code_lifting import _machine_group_fingerprints

        function_state = tuple(
            (
                str(name),
                _machine_group_fingerprints(function),
                tuple(function.metadata.get("machine_group_fingerprints", ())),
                tuple(function.metadata.get("machine_block_addresses", ())),
            )
            for name, function in request.module.functions.items()
        )
        payload = (
            function_state, str(request.name), int(request.trigger_address),
            request.stream_entry_address,
            tuple(sorted((int(k), int(v)) for k, v in request.exit_redirects.items())),
            tuple(sorted(
                (int(address), tuple(sorted(map(str, facts))))
                for address, facts in request.proven_facts_by_address.items()
            )),
            tuple(sorted(
                (int(address), repr(fields))
                for address, fields in request.encoding_fields_by_address.items()
            )),
            tuple(sorted(
                (int(address), repr(instruction))
                for address, instruction
                in request.allocated_instructions_by_address.items()
            )),
            bool(request.allow_new_selection), tuple(request.selected_blocks),
        )
        return sha256(repr(payload).encode("utf-8")).hexdigest()

    def compile(self, request: SSAWriteHeadRequest) -> RecompiledMachineStream:
        from .pe_recompilation import build_pe_recompilation_ledger

        request = replace(
            request,
            exit_redirects={
                int(source): int(target)
                for source, target in request.exit_redirects.items()
            },
            proven_facts_by_address={
                int(address): tuple(map(str, facts))
                for address, facts in request.proven_facts_by_address.items()
            },
            encoding_fields_by_address=dict(request.encoding_fields_by_address),
            allocated_instructions_by_address=dict(
                request.allocated_instructions_by_address
            ),
            selected_blocks=tuple(
                (str(function_name), str(block_name))
                for function_name, block_name in request.selected_blocks
            ),
        )
        identity = self._request_identity(request)
        cached = self._fragments.get(identity)
        if cached is not None:
            self._hits += 1
            return cached

        ledger = build_pe_recompilation_ledger(
            request.module,
            proven_facts_by_address=request.proven_facts_by_address,
            encoding_fields_by_address=request.encoding_fields_by_address,
            allocated_instructions_by_address=(
                request.allocated_instructions_by_address
            ),
            allow_new_selection=bool(request.allow_new_selection),
            selected_blocks=(
                request.selected_blocks if request.selected_blocks else None
            ),
        )
        stream = RecompiledMachineStream.from_recompilation_ledger(
            request.module, ledger, name=request.name,
            trigger_address=int(request.trigger_address),
            stream_entry_address=request.stream_entry_address,
            exit_redirects=request.exit_redirects,
        )
        if len(self._fragments) >= self.maximum_cached_fragments:
            self._fragments.pop(next(iter(self._fragments)))
        self._fragments[identity] = stream
        self._misses += 1
        return stream

    def compile_block(
        self,
        request: SSAWriteHeadRequest,
        function_name: str,
        block_name: str,
    ) -> RecompiledMachineStream:
        """Emit one existing repository-SSA block as an in-memory stream."""

        if request.selected_blocks:
            raise ValueError(
                "compile_block cannot override an already scoped write request"
            )
        return self.compile(replace(
            request,
            selected_blocks=((str(function_name), str(block_name)),),
        ))

    def compile_function(
        self,
        request: SSAWriteHeadRequest,
        function_name: str,
    ) -> RecompiledMachineStream:
        """Emit one existing SSA function, including a planned region."""

        if request.selected_blocks:
            raise ValueError(
                "compile_function cannot override an already scoped write request"
            )
        name = str(function_name)
        function = request.module.functions.get(name)
        if function is None:
            raise KeyError(f"unknown SSA function {name!r}")
        return self.compile(replace(
            request,
            selected_blocks=tuple(
                (name, str(block_name)) for block_name in function.blocks
            ),
        ))


@dataclass(frozen=True, slots=True)
class SSAExternalCodeReference:
    """Alternate SSA-owned code supplied to the write head.

    This is not a callable import.  The reference identifies repository SSA
    whose proof-complete write result supersedes the ordinary read event at its
    trigger.  Execution still receives machine instructions and machine state.
    """

    reference_id: int
    name: str
    write_request: SSAWriteHeadRequest


@dataclass(frozen=True, slots=True)
class MachineStreamExecutionEvent:
    """Proof that one executed instruction came through an injected stream."""

    stream_name: str
    route: MachineStreamRoute
    external_reference_id: int | None
    trigger_address: int | None
    source_address: int
    source_encoded: bytes
    instruction_address: int
    encoded: bytes
    token: int
    source_read_head: str
    read_head: str
    source_read_head_microsteps: int
    read_head_microsteps: int
    ssa_line_ids: tuple[str, ...]
    witness: str | None
    redirected_to: int | None
    result: MachineExecutionResult


class MachineInstructionStreamInterposer:
    """Read-head -> SSA/write-head -> executor-head instruction pipeline.

    Every route begins at the read head.  An unchanged instruction can pass
    straight to execution; an SSA-recompiled stream can replace it; or an
    SSA external code reference can supply a different compiled stream.  The
    latter is code-source injection, never an external runtime call.
    """

    def __init__(
        self,
        executor: MachineExecutionOrchestrator,
        *streams: RecompiledMachineStream,
        source_ssa: IRModule | None = None,
        write_requests: tuple[SSAWriteHeadRequest, ...] = (),
        external_code_references: tuple[SSAExternalCodeReference, ...] = (),
        write_head: BidirectionalSSAWriteHead | None = None,
    ) -> None:
        self.executor = executor
        self.write_head = write_head or BidirectionalSSAWriteHead()
        written_streams = tuple(
            self.write_head.compile(item) for item in write_requests
        )
        referenced_streams = tuple(
            self.write_head.compile(item.write_request)
            for item in external_code_references
        )
        self.streams = tuple((*streams, *written_streams, *referenced_streams))
        source_lines: dict[int, list[str]] = {}
        if source_ssa is not None:
            for function_name, function in source_ssa.functions.items():
                for block_name, block in function.blocks.items():
                    for ordinal, instruction in enumerate(block.instrs):
                        address = instruction.attributes.get("machine_address")
                        if address is not None:
                            source_lines.setdefault(int(address), []).append(
                                f"{function_name}:{block_name}:{ordinal}"
                            )
        self._source_ssa_line_ids = MappingProxyType({
            address: tuple(values) for address, values in source_lines.items()
        })
        triggers: dict[
            int, tuple[RecompiledMachineStream, int, MachineStreamRoute]
        ] = {}
        instructions: dict[int, RecompiledMachineStream] = {}
        redirects: dict[int, tuple[RecompiledMachineStream, int]] = {}
        reference_stream_ids = {id(item) for item in referenced_streams}
        reference_ids = {
            id(stream): int(reference.reference_id)
            for reference, stream in zip(external_code_references, referenced_streams)
        }
        for stream in self.streams:
            route = (
                MachineStreamRoute.EXTERNAL_SSA_REFERENCE
                if id(stream) in reference_stream_ids
                else MachineStreamRoute.SSA_RECOMPILE
            )
            for source, target in stream.triggers.items():
                if source in triggers:
                    raise ValueError(f"multiple streams trigger at {source:#x}")
                triggers[source] = (stream, target, route)
            for address in stream.instructions:
                if address in instructions:
                    raise ValueError(f"multiple streams own instruction {address:#x}")
                instructions[address] = stream
            for source, target in stream.exit_redirects.items():
                if source in redirects:
                    raise ValueError(f"multiple streams redirect exit {source:#x}")
                redirects[source] = (stream, target)
        self._triggers = MappingProxyType(triggers)
        self._instructions = MappingProxyType(instructions)
        self._redirects = MappingProxyType(redirects)
        self._stream_routes = MappingProxyType({
            id(stream): (
                MachineStreamRoute.EXTERNAL_SSA_REFERENCE
                if id(stream) in reference_stream_ids
                else MachineStreamRoute.SSA_RECOMPILE
            )
            for stream in self.streams
        })
        self._external_reference_ids = MappingProxyType(reference_ids)
        self._read_head = X86TensorReadHead.from_profile(
            controlled_x86_64_read_head_profile(),
        )
        self._decoder = X86ReferenceDecoder()
        self.last_stream_event: MachineStreamExecutionEvent | None = None

    def _read_instruction(
        self, address: int, encoded: bytes, *, require_all: bool = True,
    ) -> tuple[Any, int, str, int]:
        """Frame one instruction through the complete repository vocabulary.

        The scalar reference decoder owns complete token/framing coverage. If
        the token is also represented in the compiled tensor profile, that
        head must independently emit the same token and extent.  A mismatch is
        an error; profile absence is reported as ``reference`` rather than
        turning the accelerator's smaller table into an architecture limit.
        """

        decoded, end = self._decoder.decode_one(
            memoryview(encoded), 0, base_address=int(address),
        )
        if require_all and end != len(encoded):
            raise ValueError(
                f"replacement stream framing mismatch at {address:#x}: "
                f"consumed {end}/{len(encoded)} bytes"
            )
        if end > len(encoded) or not bytes(decoded.encoded):
            raise ValueError(f"invalid reference decode extent at {address:#x}")
        token = int(decoded.token)
        accelerated_tokens = {
            int(row.token) for row in self._read_head.profile.rows
        } if self._read_head.profile is not None else set()
        if token not in accelerated_tokens:
            return decoded, token, "reference", 0

        batch = X86ReadBatch(
            octets=AbstractTensor.get_tensor([list(encoded[:end])], dtype="int64"),
            valid_lengths=AbstractTensor.get_tensor([end], dtype="int64"),
            base_addresses=AbstractTensor.get_tensor([int(address)], dtype="int64"),
        )
        runtime = X86ReversibleReadHead.create(self._read_head, batch)
        emitted = None
        for microsteps in range(1, 65):
            state = runtime.transition()
            status = int(state.status.item())
            if status in {int(ReadStatus.EMITTED), int(ReadStatus.HALTED)}:
                emitted = state
                break
            if status == int(ReadStatus.FAILED):
                raise ValueError(
                    f"replacement stream read failed at {address:#x}: "
                    f"failure={int(state.failure.item())}"
                )
        if emitted is None:
            raise RuntimeError(f"replacement stream read did not emit at {address:#x}")
        consumed = int(emitted.cursor.item())
        if (
            consumed != end
            or consumed > len(encoded)
            or int(emitted.instruction_start.item()) != 0
        ):
            raise ValueError(
                f"replacement stream framing mismatch at {address:#x}: "
                f"consumed {consumed}/{end} reference-decoded bytes"
            )
        if int(decoded.token) != int(emitted.token.item()):
            raise ValueError(f"replacement stream token mismatch at {address:#x}")
        return decoded, int(emitted.token.item()), "tensor-verified", microsteps

    def _read_original(
        self, state: MachineExecutionState,
    ) -> tuple[Any, int, str, int]:
        """Read guest bytes before choosing pass-through or supersession."""

        address = int(state.pc)
        raw = bytearray()
        for index in range(15):
            try:
                raw.append(int(state.memory[address + index]))
            except KeyError:
                break
        if not raw:
            static = self.executor.instructions.get(address)
            if static is None:
                # Preserve the executor's exact structured blocked result.
                raise KeyError(address)
            raw.extend(static.encoded)
        return self._read_instruction(address, bytes(raw), require_all=False)

    def step(self, state: MachineExecutionState) -> MachineExecutionResult:
        """Execute one normal or stream-injected instruction transition."""

        self.last_stream_event = None
        requested_pc = int(state.pc)
        trigger = self._triggers.get(requested_pc)
        # A trigger wins over a numerically overlapping write-stream address:
        # the first read is always the original guest source.  Only feedback
        # into a non-trigger stream address reads already-written bytes.
        resident_stream = (
            None if trigger is not None else self._instructions.get(requested_pc)
        )
        trigger_address = None
        source_decoded = None
        source_token = None
        source_head = None
        source_microsteps = None
        if resident_stream is not None:
            try:
                source_decoded, source_token, source_head, source_microsteps = (
                    self._read_instruction(
                        requested_pc, resident_stream.instructions[requested_pc],
                    )
                )
            except (KeyError, ValueError, RuntimeError) as error:
                return MachineExecutionResult(
                    MachineExecutionStatus.BLOCKED_CONTROL, state,
                    f"read head could not frame written stream instruction at "
                    f"{requested_pc:#x}: {error}",
                )
        else:
            try:
                source_decoded, source_token, source_head, source_microsteps = self._read_original(state)
            except (KeyError, ValueError, RuntimeError) as error:
                return MachineExecutionResult(
                    MachineExecutionStatus.BLOCKED_CONTROL, state,
                    f"read head could not frame guest instruction at "
                    f"{requested_pc:#x}: {error}",
                )
        if trigger is not None:
            stream, stream_pc, route = trigger
            trigger_address = requested_pc
            state = replace(state, pc=int(stream_pc))
        else:
            stream = resident_stream
        if stream is None:
            if source_decoded is None:
                return self.executor.step(state)
            result = self.executor.execute_decoded(state, source_decoded)
            self.last_stream_event = MachineStreamExecutionEvent(
                "guest-pass-through", MachineStreamRoute.PASS_THROUGH,
                None, None, requested_pc,
                bytes(source_decoded.encoded), requested_pc,
                bytes(source_decoded.encoded), int(source_token),
                str(source_head), str(source_head),
                int(source_microsteps), int(source_microsteps),
                self._source_ssa_line_ids.get(requested_pc, ()),
                None, None, result,
            )
            return result
        if trigger is None:
            route = self._stream_routes[id(stream)]
        address = int(state.pc)
        encoded = stream.instructions.get(address)
        if encoded is None:
            # A call from the injected stream is executing ordinary guest code.
            return self.executor.step(state)
        if resident_stream is stream and source_decoded is not None:
            decoded, token, head_name, microsteps = (
                source_decoded, int(source_token), str(source_head),
                int(source_microsteps)
            )
        else:
            # Trigger supersession reads the write-head output as a second
            # framing event before the executor sees it.
            decoded, token, head_name, microsteps = self._read_instruction(address, encoded)
        result = self.executor.execute_decoded(state, decoded)
        redirected_to = None
        redirect = self._redirects.get(int(result.state.pc))
        if redirect is not None and redirect[0] is stream:
            redirected_to = int(redirect[1])
            result = replace(result, state=replace(result.state, pc=redirected_to))
        self.last_stream_event = MachineStreamExecutionEvent(
            stream.name, route, self._external_reference_ids.get(id(stream)),
            trigger_address, requested_pc,
            (
                b"" if source_decoded is None
                else bytes(source_decoded.encoded)
            ),
            address, encoded, token,
            ("reference" if source_head is None else str(source_head)), head_name,
            (0 if source_microsteps is None else int(source_microsteps)),
            microsteps,
            stream.ssa_line_ids.get(address, ()), stream.witnesses.get(address),
            redirected_to, result,
        )
        return result

    def owns_address(self, address: int) -> bool:
        """Whether the write stream, rather than ordinary guest code, owns RIP."""

        value = int(address)
        return value in self._triggers or value in self._instructions

    def __getattr__(self, name: str) -> Any:
        return getattr(self.executor, name)


class MachineStreamBlockDispatcher:
    """Commit injected AMD64 streams through the existing runner protocol.

    At a stream trigger or replacement address, the AMD64 read/write-head
    pipeline owns execution.  Everywhere else an optional existing compiled
    dispatcher (currently the repository's Wasm block dispatcher) gets the
    same request.  Returning ``None`` leaves the runner's established
    translated/pass-through path in control.  Every successful stream step is
    committed as one normal reversible machine edge.
    """

    def __init__(
        self,
        interposer: MachineInstructionStreamInterposer,
        *,
        fallback_dispatcher: Any | None = None,
    ) -> None:
        self.interposer = interposer
        self.fallback_dispatcher = fallback_dispatcher
        self.stream_executions = 0
        self.committed_instructions = 0
        self.last_events: tuple[MachineStreamExecutionEvent, ...] = ()

    @property
    def statistics(self) -> Mapping[str, int]:
        fallback = getattr(self.fallback_dispatcher, "statistics", {})
        return MappingProxyType({
            "stream_executions": self.stream_executions,
            "stream_committed_instructions": self.committed_instructions,
            **{f"fallback_{key}": int(value) for key, value in fallback.items()},
        })

    def execute(
        self,
        core: Any,
        maximum_instructions: int,
        *,
        transition_observer: Any | None = None,
    ) -> tuple[MachineExecutionResult, ...] | None:
        limit = int(maximum_instructions)
        if limit <= 0 or core.state.halted:
            return None
        if core.executor is not self.interposer.executor:
            raise ValueError(
                "machine stream dispatcher and reversible core must share an executor"
            )
        if not self.interposer.owns_address(core.state.pc):
            if self.fallback_dispatcher is None:
                return None
            return self.fallback_dispatcher.execute(
                core, limit, transition_observer=transition_observer,
            )
        results: list[MachineExecutionResult] = []
        events: list[MachineStreamExecutionEvent] = []
        for _ in range(limit):
            if not self.interposer.owns_address(core.state.pc):
                break
            result = self.interposer.step(core.state)
            event = self.interposer.last_stream_event
            core.commit_execution_result(result)
            results.append(result)
            if event is not None:
                events.append(event)
            if transition_observer is not None:
                transition_observer()
            if result.status is not MachineExecutionStatus.RUNNING:
                break
        self.last_events = tuple(events)
        if results:
            self.stream_executions += 1
            self.committed_instructions += len(results)
        return tuple(results) or None

    def close(self) -> None:
        close = getattr(self.fallback_dispatcher, "close", None)
        if close is not None:
            close()


__all__ = [
    "BidirectionalSSAWriteHead", "MachineInstructionStreamInterposer",
    "MachineStreamBlockDispatcher",
    "MachineStreamExecutionEvent",
    "MachineStreamRoute", "RecompiledMachineStream",
    "SSAExternalCodeReference", "SSAWriteHeadRequest",
]
