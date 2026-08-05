"""Prove the real per-instruction dispatch compiles, not just isolated helpers.

MachineExecutionOrchestrator.step() is the actual coordinator loop's entry
point: decode the instruction at state.pc, dispatch on its semantic token,
apply effects, advance. This is the thing CMD_BINARY_EXECUTOR_COMPILATION_HANDOFF.md
calls "the compiled machine coordinator ABI" -- not one arbitrary function
from the executor, the dispatch loop itself. These tests compile it
directly through compile_ast_aot with real MachineExecutionOrchestrator
fixtures (not mocks), the same way every other AOT-compile test in this
session was verified.
"""

from __future__ import annotations

import inspect
import textwrap
from types import SimpleNamespace

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.compiler import machine_execution as me
from src.compiler.amd64_machine_semantics import condition_holds, default_effect_handlers
from src.compiler.machine_execution import MachineExecutionOrchestrator, MachineExecutionState
from src.compiler.machine_reference_vocabulary import (
    MachineSemanticToken, RelativeAddressOperand,
)

_STEP_SOURCE = textwrap.dedent(inspect.getsource(MachineExecutionOrchestrator.step))
_STEP_BINDINGS = {
    "MachineExecutionResult": me.MachineExecutionResult,
    "MachineExecutionStatus": me.MachineExecutionStatus,
}


def test_step_dispatch_compiles_for_a_plain_instruction():
    instruction = SimpleNamespace(
        address=0x2000, encoded=b"\x90",
        semantic=MachineSemanticToken.NO_OPERATION,
        token=SimpleNamespace(name="NOP"), operands=(),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x2000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(instructions=(instruction,))),),
    )
    executor = MachineExecutionOrchestrator(program, effect_handlers=default_effect_handlers())
    state = MachineExecutionState(pc=0x2000)

    compilation = compile_ast_aot(
        _STEP_SOURCE, "step", {"self": executor, "state": state},
        python_bindings=_STEP_BINDINGS, precompile_only=True,
    )
    assert compilation.control_shortfalls == ()


def test_step_dispatch_compiles_through_a_conditional_branch():
    # Exercises _step_decoded's CONDITIONAL_RELATIVE_JUMP branch, which
    # calls self.predicate_handler(state, instruction) -- a call into a
    # second real function (condition_holds), transparently traced the
    # same way _as_memory was inside complete_external_call_state earlier
    # in this investigation.
    instruction = SimpleNamespace(
        address=0x2000, encoded=b"\x75\x00",
        semantic=MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        token=SimpleNamespace(name="JNE_REL8"),
        operands=(RelativeAddressOperand(displacement=0, width=8, target_address=0x2100),),
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x2000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(instructions=(instruction,))),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(), predicate_handler=condition_holds,
    )
    state = MachineExecutionState(pc=0x2000)

    compilation = compile_ast_aot(
        _STEP_SOURCE, "step", {"self": executor, "state": state},
        python_bindings=_STEP_BINDINGS, precompile_only=True,
    )
    assert compilation.control_shortfalls == ()
