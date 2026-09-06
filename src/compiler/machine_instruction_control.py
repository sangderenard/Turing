"""Legacy breakpoint and mapped-byte patch control.

.. deprecated::
   This module changes guest memory and therefore is not executor-stream
   interposition. New read/edit/write/execute work must use
   :mod:`machine_stream_interposition`, where original and written bytes pass
   through the tensor read head and the guest image remains unchanged.

Breakpoint/hook/patch/resume control sits on top of the reversible executor.

This adds no new execution semantics to ``ReversibleMachineExecutor`` or
``MachineExecutionOrchestrator`` -- it is a thin driver built entirely from
existing primitives, verified to already exist for this purpose:

- Stopping at a program counter: nothing in the executor stops early on its
  own, so ``run_with_hook`` steps one instruction at a time and checks
  ``state.pc`` itself before each step, the same way a caller-supplied
  ``transition_observer`` would.
- Non-destructive lookahead: ``ReversibleMachineExecutor.fork()`` already
  gives an independent, cheap copy of history sharing the same
  ``executor`` (decode/semantics) but none of the original's mutable state
  list -- ``peek_ahead`` steps a fork and simply discards it.
- Overriding state: ``ReversibleMachineExecutor.commit_shell_effect()``
  already exists exactly for "journal shell-owned device input without
  completing a guest call" -- the same mechanism used elsewhere (thread-wait
  bookkeeping in ``binary_machine_program.py``) to reversibly replace the
  current state with an edited one.
- Patching mapped binary instructions: ``MachineExecutionOrchestrator._decode_instruction_from_state``
  (machine_execution.py:582) already prefers a *dynamic* decode straight
  from ``state.memory`` at the target address whenever the static
  ``self.instructions`` table doesn't have a byte-matching entry there --
  this is the same path ordinary self-modifying/JIT'd guest code already
  goes through. This legacy API writes the desired
  bytes into guest memory (reversibly, via a new committed state) at an
  address the executor already considers executable. It does not attempt
  to fabricate a *new* executable region -- ``VirtualMemoryEffect`` only
  models allocate/release, not a protection change on an existing region,
  so making previously-non-executable memory executable is out of scope
  here and raises rather than silently pretending to support it.
- Resuming: either at the exact point already reached (nothing special
  needed -- the loop just continues), or at an arbitrary address (a plain
  ``commit_shell_effect(replace(state, pc=...))``, the same reversible
  "control transfer via a new committed state" other completions use).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from .amd64_machine_semantics import PagedByteMemory
from .machine_execution import (
    MachineExecutionResult, MachineExecutionState, MachineExecutionStatus,
    ReversibleMachineExecutor,
)


class InstructionPatchError(ValueError):
    """A requested instruction patch targets memory the executor cannot run."""


@dataclass(frozen=True)
class HookResult:
    """What a breakpoint hook wants done before execution continues.

    ``override_state`` replaces the current state outright (already fully
    constructed by the caller). ``patch`` is a sequence of
    ``(address, encoded_bytes)`` pairs applied in order, each via
    ``patch_instruction_bytes``. ``resume_pc`` retargets the program
    counter after any override/patch -- "resume at some point on the
    original program" that need not be where execution stopped.
    ``stop`` ends ``run_with_hook``'s loop without executing another
    instruction.
    """

    override_state: MachineExecutionState | None = None
    patch: tuple[tuple[int, bytes], ...] = ()
    resume_pc: int | None = None
    stop: bool = False


def peek_ahead(
    reversible: ReversibleMachineExecutor, distance: int,
) -> tuple[MachineExecutionState, tuple[MachineExecutionResult, ...]]:
    """Look ``distance`` real instructions ahead without touching ``reversible``.

    Forks (cheap: shares the immutable ``executor``, copies only the
    resident state/edge lists up to the current position), steps the fork,
    and lets it be discarded. The original's history and position are
    untouched -- verified by tests, not merely implied by ``fork()``'s
    docstring.
    """

    if distance <= 0:
        raise ValueError("peek_ahead distance must be positive")
    scout = reversible.fork()
    results = scout.step_block_forward(distance)
    return scout.state, results


def _executable(reversible: ReversibleMachineExecutor, address: int) -> bool:
    executor = reversible.executor
    if address // 4096 in executor._executable_pages:
        return True
    virtual_memory = reversible.state.virtual_memory
    return virtual_memory is not None and virtual_memory.is_executable(address)


def patch_instruction_bytes(
    reversible: ReversibleMachineExecutor,
    address: int,
    encoded: bytes,
) -> MachineExecutionState:
    """Write ``encoded`` into guest memory at ``address`` and commit it.

    Requires ``address`` to already be executable (statically known or
    marked executable in ``virtual_memory``) -- this writes bytes for the
    existing dynamic-decode path to pick up on the next fetch there, it does
    not grant execute permission to memory that lacks it. Every destination
    byte must already be mapped, matching the same-page-must-exist
    discipline ``complete_external_call_state`` uses for external memory
    writes -- a patch cannot manufacture a brand new guest mapping either.
    """

    if not encoded:
        raise ValueError("instruction patch bytes must be non-empty")
    if not _executable(reversible, address):
        raise InstructionPatchError(
            f"address {address:#x} is not in an executable region; "
            "patch_instruction_bytes only overwrites bytes in memory the "
            "executor already treats as code, it does not grant execute "
            "permission to a new region"
        )
    state = reversible.state
    memory = state.memory
    for index in range(len(encoded)):
        try:
            memory[address + index]
        except KeyError as error:
            raise InstructionPatchError(
                f"address {address + index:#x} is not mapped guest memory"
            ) from error
    memory = memory.map_bytes(address, encoded)
    return reversible.commit_shell_effect(replace(state, memory=memory))


def resume_at(
    reversible: ReversibleMachineExecutor, address: int,
) -> MachineExecutionState:
    """Retarget the program counter to ``address`` and commit the transfer.

    Ordinary reversible control transfer -- the same "new committed state"
    shape ``complete_external_call_state`` uses for a completion's ``pc``
    update, just triggered by a hook instead of a serviced external call.
    """

    return reversible.commit_shell_effect(replace(reversible.state, pc=int(address)))


def run_with_hook(
    reversible: ReversibleMachineExecutor,
    *,
    breakpoints: frozenset[int] | set[int],
    hook: Callable[[MachineExecutionState, ReversibleMachineExecutor], HookResult],
    maximum_transitions: int = 1_000_000,
) -> tuple[MachineExecutionResult, ...]:
    """Step forward, running ``hook`` each time ``state.pc`` hits a breakpoint.

    The hook runs *before* the breakpointed instruction executes (ordinary
    debugger semantics: the breakpoint is a pre-instruction stop). Its
    ``HookResult`` may override state, patch bytes into guest memory,
    retarget ``pc`` to resume somewhere other than the breakpoint, and/or
    stop the run outright. Everything the hook does is committed through
    ``commit_shell_effect``/``patch_instruction_bytes``, so it stays fully
    reversible: ``reversible.step_backward()`` walks back through hook
    effects exactly like ordinary instruction steps.
    """

    breakpoints = frozenset(int(item) for item in breakpoints)
    results: list[MachineExecutionResult] = []
    for _ in range(maximum_transitions):
        if reversible.state.pc in breakpoints:
            outcome = hook(reversible.state, reversible)
            if outcome.override_state is not None:
                reversible.commit_shell_effect(outcome.override_state)
            for address, encoded in outcome.patch:
                patch_instruction_bytes(reversible, address, encoded)
            if outcome.resume_pc is not None:
                resume_at(reversible, outcome.resume_pc)
            if outcome.stop:
                break
        result = reversible.step_forward()
        results.append(result)
        if result.status is not MachineExecutionStatus.RUNNING:
            break
    return tuple(results)


__all__ = [
    "HookResult",
    "InstructionPatchError",
    "patch_instruction_bytes",
    "peek_ahead",
    "resume_at",
    "run_with_hook",
]
