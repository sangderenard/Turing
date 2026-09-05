from dataclasses import replace

import pytest

from examples.reversible_demo_subject import build_reversible_demo_subject
from examples.reversible_machine_web_host import _new_machine_bytes
from src.compiler.machine_execution import MachineExecutionStatus
from src.compiler.machine_instruction_control import (
    HookResult, InstructionPatchError, patch_instruction_bytes, peek_ahead,
    resume_at, run_with_hook,
)

# DEMO_ENTRY_CODE = mov rax,42 (7B) ; add rax,1 (4B) ; nop (1B) ; ret (1B)
ADD_RAX_1 = bytes.fromhex("48 83 c0 01")
ADD_RAX_2 = bytes.fromhex("48 83 c0 02")


def _machine():
    subject = build_reversible_demo_subject()
    return _new_machine_bytes(subject, {}, "translated")


def test_breakpoint_hook_fires_exactly_once_before_the_target_instruction():
    machine = _machine()
    try:
        core = machine.machine.cores[0]
        entry = core.state.pc
        seen = []

        def hook(state, reversible):
            seen.append((state.pc, state.registers[0]))
            return HookResult()

        results = run_with_hook(
            core, breakpoints={entry + 7}, hook=hook, maximum_transitions=3,
        )
        assert seen == [(entry + 7, 42)]  # mov already ran; add has not
        assert len(results) == 3
        assert core.state.registers[0] == 43  # mov 42, add 1 -> 43, nop
        assert core.state.pc == entry + 12
    finally:
        machine.close()


def test_peek_ahead_does_not_touch_the_original_executor():
    machine = _machine()
    try:
        core = machine.machine.cores[0]
        entry = core.state.pc
        original_position = core.position

        scout_state, results = peek_ahead(core, 2)

        assert scout_state.registers[0] == 43
        assert scout_state.pc == entry + 11
        assert len(results) == 2
        # The original is completely untouched.
        assert core.position == original_position
        assert core.state.pc == entry
        assert core.state.registers[0] == 0
    finally:
        machine.close()


def test_hook_can_override_state_before_the_breakpointed_instruction_runs():
    machine = _machine()
    try:
        core = machine.machine.cores[0]
        entry = core.state.pc

        def hook(state, reversible):
            registers = (999,) + state.registers[1:]
            return HookResult(override_state=replace(state, registers=registers))

        run_with_hook(core, breakpoints={entry + 7}, hook=hook, maximum_transitions=2)
        # The override took effect, and "add rax, 1" then ran on top of it:
        # 999 + 1 = 1000, not 43. If the override were ignored this would
        # still read 43.
        assert core.state.registers[0] == 1000
        assert core.state.pc == entry + 11
    finally:
        machine.close()


def test_hook_can_patch_the_breakpointed_instructions_bytes():
    machine = _machine()
    try:
        core = machine.machine.cores[0]
        entry = core.state.pc

        def hook(state, reversible):
            return HookResult(patch=((entry + 7, ADD_RAX_2),))

        run_with_hook(core, breakpoints={entry + 7}, hook=hook, maximum_transitions=2)
        # The patched "add rax, 2" ran instead of the original "add rax, 1".
        assert core.state.registers[0] == 44
    finally:
        machine.close()


def test_patch_instruction_bytes_rejects_a_non_executable_address():
    machine = _machine()
    try:
        core = machine.machine.cores[0]
        with pytest.raises(InstructionPatchError):
            patch_instruction_bytes(core, 0x9999999000, ADD_RAX_2)
    finally:
        machine.close()


def test_hook_can_resume_at_a_different_point_skipping_the_breakpointed_instruction():
    machine = _machine()
    try:
        core = machine.machine.cores[0]
        entry = core.state.pc

        def hook(state, reversible):
            return HookResult(resume_pc=entry + 11)  # skip straight to the nop

        # transition 1: mov rax, 42 (reaches the breakpoint at entry + 7)
        # transition 2: hook resumes at entry + 11 (the nop), which then runs
        run_with_hook(core, breakpoints={entry + 7}, hook=hook, maximum_transitions=2)
        assert core.state.pc == entry + 12  # past the nop; "add" was skipped
        assert core.state.registers[0] == 42  # add rax, 1 never ran
    finally:
        machine.close()


def test_run_with_hook_effects_are_fully_reversible():
    machine = _machine()
    try:
        core = machine.machine.cores[0]
        entry = core.state.pc

        def hook(state, reversible):
            return HookResult(patch=((entry + 7, ADD_RAX_2),))

        run_with_hook(core, breakpoints={entry + 7}, hook=hook, maximum_transitions=2)
        assert core.state.registers[0] == 44

        core.step_backward()  # undo the patched add
        assert core.state.registers[0] == 42
        core.step_backward()  # undo the patch-memory-write commit
        core.step_backward()  # undo the mov
        assert core.state.registers[0] == 0
        assert core.state.pc == entry
    finally:
        machine.close()


def test_resume_at_is_an_ordinary_reversible_control_transfer():
    machine = _machine()
    try:
        core = machine.machine.cores[0]
        entry = core.state.pc
        core.step_forward()  # mov rax, 42
        resume_at(core, entry + 11)  # jump straight to the nop
        assert core.state.pc == entry + 11
        assert core.state.registers[0] == 42

        core.step_backward()
        assert core.state.pc == entry + 7  # back to right after the mov
    finally:
        machine.close()
