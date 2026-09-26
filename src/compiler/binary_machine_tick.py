"""One repeated coordinating state tick around the real binary machine.

Everything inside stays ordinary Python: a real ``BinaryMachineProgram``,
real objects, real method calls. Nothing here is a second implementation of
decode or execution semantics -- ``BinaryMachineProgram.load_pe`` and
``machine.runner.tick`` are the exact same calls the native host already
uses, verified against the real demo subject before this was written this
way (``rax`` 42 -> 43 across three real ticks). This is the outer
coordinating step, not the inner one: ``MachineExecutionOrchestrator.step``
(called once per admitted instruction, inside ``runner.tick``) already
exists and is not rewritten here. This module answers a different question
every call: given the whole state right now, what does this tick need to
do -- decode, run some bounded number of instructions, or wait -- before
handing the tick back.

The boundary is flat (a hash token, a phase, a bounded instruction budget,
the resulting register/pc/step state) the same way every other compiled
entrypoint in this codebase is: a caller supplies the previous tick's
outputs as this tick's inputs, nothing else required. What is *not* flat is
deliberately not flattened -- the live ``BinaryMachineProgram`` stays a real
Python object, held between calls by the same file-hash key a caller's
``phase``/``program_hash`` fields already name, so "decide every tick
whether to repopulate the decoding" is a real cache lookup, not a comment.
"""

from __future__ import annotations

import hashlib

from .binary_machine_program import BinaryMachineProgram
from .machine_state_buffer import MachineRunDirection

PHASE_EMPTY = 0
PHASE_DECODE = 1
PHASE_RUN = 2

DEFAULT_MAXIMUM_FILE_SIZE = 128 * 1024 * 1024

# Keyed by the exact loaded bytes' hash, which is also the caller-visible
# ``program_hash`` field -- the same identity, not two. A tick never
# re-decodes a file it has already decoded; it reuses the live machine and
# advances it in place, the same way the native host would keep running one
# process rather than reconstructing it every tick.
_live_machines: dict[str, BinaryMachineProgram] = {}


def _file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _cached_machine(key: str) -> BinaryMachineProgram | None:
    return _live_machines.get(key)


def _cache_machine(key: str, machine: BinaryMachineProgram, *, replace_all: bool) -> None:
    # A plain ``dict[key] = value`` assignment inside traced source is
    # lowered as a tensor indexed-store (clone-then-write); this dict holds
    # live Python objects, not tensor data, so the write has to happen
    # somewhere the compiler never traces into -- exactly like every other
    # opaque helper here. ``replace_all`` mirrors the two call sites' two
    # real behaviors (a genuinely new file evicts the old machine; a
    # same-process cache miss for an already-decoded hash does not).
    if replace_all:
        _live_machines.clear()
    _live_machines[key] = machine


def tick(
    phase: int,
    file_bytes: bytes,
    file_size: int,
    program_hash: str,
    instructions_per_tick: int,
    host_events: tuple = (),
):
    """Advance exactly one admitted clock tick for the whole program.

    ``phase``/``program_hash`` are state like any other field, carried
    forward by the caller the same way register contents are:

    - ``PHASE_EMPTY``: no file loaded yet; wait.
    - ``PHASE_DECODE``: the loaded bytes' hash does not match
      ``program_hash`` (a new file, or the first one) -- decode it via the
      real ``BinaryMachineProgram.load_pe`` and report the entry state.
      Real work, done once per distinct file, not once per tick.
    - ``PHASE_RUN``: the hash already matches a live, decoded machine.
      Apply any pending host events, then run up to
      ``instructions_per_tick`` real instructions through the real
      orchestrator (via ``machine.runner.tick``) before handing the tick
      back -- the dt-budget the whole point of an outer tick is to enforce,
      not the inner step's concern.

    ``host_events`` is the communication channel for anything the shell
    needs the machine to see this tick that is not itself state -- a file
    write, an injected device byte stream. Each entry is
    ``(kind, *arguments)``; unrecognized kinds are rejected, not ignored,
    so a caller finds out immediately if this tick's event vocabulary does
    not yet cover what it sent.
    """

    current_hash = _file_hash(bytes(file_bytes[:file_size])) if file_size else None

    if phase == PHASE_EMPTY:
        if current_hash is None:
            return _report(PHASE_EMPTY, program_hash, None)
        return _report(PHASE_DECODE, program_hash, None)

    if phase == PHASE_DECODE or current_hash != program_hash:
        if current_hash is None:
            return _report(PHASE_EMPTY, "", None)
        machine = BinaryMachineProgram.load_pe(
            bytes(file_bytes[:file_size]),
            maximum_file_size=DEFAULT_MAXIMUM_FILE_SIZE,
        )
        _cache_machine(current_hash, machine, replace_all=True)
        core = machine.machine.cores[0]
        return _report(PHASE_RUN, current_hash, core)

    machine = _cached_machine(current_hash)
    if machine is None:
        # The hash matches what the caller last saw, but this process does
        # not hold that machine live (a fresh process, an evicted cache) --
        # this is exactly the same "the decoding needs repopulating" case
        # as a changed hash, just discovered a different way.
        machine = BinaryMachineProgram.load_pe(
            bytes(file_bytes[:file_size]),
            maximum_file_size=DEFAULT_MAXIMUM_FILE_SIZE,
        )
        _cache_machine(current_hash, machine, replace_all=False)

    for event in host_events:
        _apply_host_event(machine, event)

    core = machine.machine.cores[0]
    machine.set_direction(MachineRunDirection.FORWARD)
    budget = max(0, int(instructions_per_tick))
    completed = machine.runner.tick(budget) if budget else 0
    next_phase = PHASE_EMPTY if core.state.halted else PHASE_RUN
    return _report(next_phase, current_hash, core, completed=completed)


def _apply_host_event(machine: BinaryMachineProgram, event: tuple) -> None:
    kind = event[0]
    if kind == "device_bytes":
        _, device, data = event
        machine.inject_device_bytes(device, data)
        return
    raise ValueError(f"tick does not yet handle host event kind {kind!r}")


def _report(phase: int, program_hash: str, core, *, completed: int = 0):
    if core is None:
        return {
            "phase": phase,
            "program_hash": program_hash,
            "pc": 0,
            "registers": (0,) * 16,
            "steps": 0,
            "halted": 0,
            "completed_this_tick": completed,
        }
    return {
        "phase": phase,
        "program_hash": program_hash,
        "pc": int(core.state.pc),
        "registers": tuple(int(value) for value in core.state.registers),
        "steps": int(core.state.steps),
        "halted": int(core.state.halted),
        "completed_this_tick": completed,
    }


__all__ = ["PHASE_EMPTY", "PHASE_DECODE", "PHASE_RUN", "tick"]
