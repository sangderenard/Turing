"""Forward previously-recorded external-call completions from a system tape.

This is the "web runtime tape forwarding" requirement from
CMD_BINARY_EXECUTOR_COMPILATION_HANDOFF.md items 5-6: at a boundary the
browser-resident CapabilityGatedExternalPort cannot service live (no
registered handler for that capability yet), a TapeForwardingExternalPort
instead looks for one prior MachineSystemTape record where an identically
identified request (same reference library/symbol and same arguments) was
already serviced, and derives a completion from the exact recorded state
delta -- provenance-bound to that specific prior interaction, never guessed.

Scope: only the ``result`` register (rax) and byte-level ``memory_writes``
are reconstructed -- the same narrow completion shape
machine_system_ports.py's own simplest handlers (``_return_value``,
``_write_u64``) already produce. This module does not attempt to generically
diff every MachineExecutionState effect domain (filesystem, registry,
environment, device, thread spawns, ...); reconstructing those from a raw
state diff without knowing which effect produced them would risk silently
fabricating an effect this module cannot verify, which
CMD_BINARY_EXECUTOR_COMPILATION_HANDOFF.md explicitly forbids ("Never
fabricate success to keep the prompt moving"). A request whose recorded
completion touched any of those wider domains is not matched here; it is
left unforwarded so the caller's normal not-serviceable path applies.
"""

from __future__ import annotations

from dataclasses import dataclass

from .machine_execution import (
    MachineExecutionState, MachineExternalCallCompletion,
    MachineExternalCallRequest, MachineExternalMemoryWrite,
)
from .machine_system_ports import CapabilityGatedExternalPort
from .machine_system_tape import MachineSystemTape

_RAX = 0  # MachineExecutionState.REGISTER_NAMES[0]


def _request_identity(
    request: MachineExternalCallRequest,
) -> tuple[str, str, tuple[int, ...]]:
    reference = request.reference
    return (
        reference.library.casefold(),
        reference.symbol.casefold(),
        tuple(request.arguments) + tuple(request.stack_arguments),
    )


def _is_narrow_completion(before: MachineExecutionState, after: MachineExecutionState) -> bool:
    """Only forward completions whose recorded effect was rax + memory."""

    return (
        before.virtual_filesystem == after.virtual_filesystem
        and before.virtual_registry == after.virtual_registry
        and before.virtual_memory == after.virtual_memory
        and before.environment_state == after.environment_state
        and before.text_state == after.text_state
        and before.device_state == after.device_state
        and before.call_stack == after.call_stack
        and before.system_state == after.system_state
        and before.registers[1:] == after.registers[1:]
        and before.halted == after.halted
        and before.termination_requested == after.termination_requested
    )


def _changed_memory_writes(
    before: MachineExecutionState, after: MachineExecutionState,
) -> tuple[MachineExternalMemoryWrite, ...]:
    changed = sorted(
        address for address in set(before.memory) | set(after.memory)
        if before.memory.get(address, 0) != after.memory.get(address, 0)
    )
    writes: list[MachineExternalMemoryWrite] = []
    index = 0
    while index < len(changed):
        start = changed[index]
        end = start
        while index + 1 < len(changed) and changed[index + 1] == end + 1:
            index += 1
            end = changed[index]
        writes.append(MachineExternalMemoryWrite(
            start,
            bytes(after.memory.get(address, 0) for address in range(start, end + 1)),
        ))
        index += 1
    return tuple(writes)


def find_recorded_completion(
    tape: MachineSystemTape,
    request: MachineExternalCallRequest,
    *,
    core: int = 0,
) -> MachineExternalCallCompletion | None:
    """Find one prior tape record whose request matches ``request``'s identity.

    Returns ``None`` when no matching record exists, or when the matching
    record's recorded effect is wider than the narrow rax+memory shape this
    module reconstructs.
    """

    identity = _request_identity(request)
    for record in tape.records:
        if record.get("event") != "external_completion" or int(record["core"]) != core:
            continue
        before_sequence = record.get("parent_sequence")
        if before_sequence is None:
            continue
        before_state = tape.resume_state(core, sequence=int(before_sequence))
        matched = next(
            (
                item for item in before_state.external_requests
                if _request_identity(item) == identity
            ),
            None,
        )
        if matched is None:
            continue
        after_state = tape.resume_state(core, sequence=int(record["sequence"]))
        if not _is_narrow_completion(before_state, after_state):
            continue
        return MachineExternalCallCompletion(
            request_id=request.request_id,
            result=int(after_state.registers[_RAX]),
            memory_writes=_changed_memory_writes(before_state, after_state),
        )
    return None


@dataclass(frozen=True)
class TapeForwardingExternalPort:
    """Consult ``live`` first; fall back to an exact tape-recorded completion.

    Same ``handle``/``supports``/``wait_kind`` shape as
    CapabilityGatedExternalPort so it can replace one anywhere that consumes
    it (for example ``MachineExecutionOrchestrator.service_external_requests``)
    without that caller needing to know which source served the request.
    """

    tape: MachineSystemTape
    live: CapabilityGatedExternalPort | None = None
    core: int = 0

    def handle(
        self, request: MachineExternalCallRequest, state: MachineExecutionState,
    ) -> MachineExternalCallCompletion | None:
        if self.live is not None:
            completion = self.live.handle(request, state)
            if completion is not None:
                return completion
        return find_recorded_completion(self.tape, request, core=self.core)

    def supports(self, request: MachineExternalCallRequest) -> bool:
        if self.live is not None and self.live.supports(request):
            return True
        return find_recorded_completion(self.tape, request, core=self.core) is not None

    def wait_kind(self, request: MachineExternalCallRequest) -> str | None:
        return None if self.live is None else self.live.wait_kind(request)


__all__ = [
    "TapeForwardingExternalPort",
    "find_recorded_completion",
]
