"""Compiled binary-machine program interior.

The implementation is authored in Python and enters the ordinary compiler as
program source.  It is not a packaged Python interpreter. Runtime subject
bytes remain in the register-aware machine pipeline and never become cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .machine_chip_layout import (
    FixedProgramCacheLayout,
    RegisterBankLayout,
    build_fixed_program_cache_layout,
    build_register_bank_layout,
)
from .machine_execution import (
    MachineEffectHandler,
    MachineExecutionOrchestrator,
    MachineIndirectTargetHandler,
    MachinePredicateHandler,
    MachineVirtualMulticore,
    MachineExternalCallRequest,
    MachineExternalReference,
)
from .amd64_machine_semantics import (
    build_external_references,
    build_initial_machine_state,
    complete_external_call_state,
    condition_holds,
    default_effect_handlers,
    indirect_target,
)
from .machine_program_graph import raise_pe_to_token_multigraph
from .machine_state_buffer import (
    ExternalMachineClock,
    FreeRunningMachineRunner,
    MachineRunDirection,
    MachineSnapshotLayout,
    MachineSnapshotTripleBuffer,
    SubjectOutputBuffer,
)


class SubjectDeviceBuffers:
    """Atomically replaceable subject outputs sampled at snapshot publication."""

    def __init__(self) -> None:
        self._outputs: tuple[SubjectOutputBuffer, ...] = ()

    def publish(self, outputs: Sequence[SubjectOutputBuffer]) -> None:
        self._outputs = tuple(outputs)

    def snapshot(self) -> tuple[SubjectOutputBuffer, ...]:
        return self._outputs


@dataclass(slots=True)
class BinaryMachineProgram:
    """One loaded subject, its clocked reversible cores, and observation ABI."""

    program: object
    machine: MachineVirtualMulticore
    register_layout: RegisterBankLayout
    cache_layout: FixedProgramCacheLayout
    snapshots: MachineSnapshotTripleBuffer
    runner: FreeRunningMachineRunner
    clock: ExternalMachineClock
    devices: SubjectDeviceBuffers
    external_references: tuple[MachineExternalReference, ...]

    @classmethod
    def from_program(
        cls,
        program,
        *,
        core_count: int = 1,
        transitions_per_second: float = 60.0,
        maximum_transitions_per_tick: int = 1_000_000,
        maximum_outputs: int = 4,
        maximum_output_bytes: int = 4 * 1024 * 1024,
        effect_handlers: Mapping[int, MachineEffectHandler] | None = None,
        predicate_handler: MachinePredicateHandler | None = None,
        indirect_target_handler: MachineIndirectTargetHandler | None = None,
    ) -> "BinaryMachineProgram":
        using_default_machine = (
            effect_handlers is None
            and predicate_handler is None
            and indirect_target_handler is None
        )
        if using_default_machine:
            effect_handlers = default_effect_handlers()
            predicate_handler = condition_holds
            indirect_target_handler = indirect_target
        external_references = (
            build_external_references(program) if using_default_machine else ()
        )
        references_by_target = {
            reference.target_address: reference for reference in external_references
        }
        executor = MachineExecutionOrchestrator(
            program,
            effect_handlers=effect_handlers,
            predicate_handler=predicate_handler,
            indirect_target_handler=indirect_target_handler,
            external_target_resolver=references_by_target.get,
        )
        initial_states = None
        if using_default_machine and getattr(program.image, "encoded", None) is not None:
            initial = build_initial_machine_state(
                program, external_references=external_references,
            )
            initial_states = tuple(initial for _ in range(core_count))
        machine = MachineVirtualMulticore.create(
            executor, core_count=core_count, initial_states=initial_states,
        )
        registers = build_register_bank_layout(core_count)
        snapshot_layout = MachineSnapshotLayout.build(
            registers,
            core_count=core_count,
            maximum_outputs=maximum_outputs,
            maximum_output_bytes=maximum_output_bytes,
        )
        snapshots = MachineSnapshotTripleBuffer(snapshot_layout, registers)
        devices = SubjectDeviceBuffers()
        runner = FreeRunningMachineRunner(
            machine, snapshots, output_provider=devices.snapshot,
        )
        return cls(
            program=program,
            machine=machine,
            register_layout=registers,
            cache_layout=build_fixed_program_cache_layout(
                program, base_offset=snapshot_layout.byte_size,
            ),
            snapshots=snapshots,
            runner=runner,
            clock=ExternalMachineClock(
                transitions_per_second=transitions_per_second,
                maximum_transitions_per_tick=maximum_transitions_per_tick,
            ),
            devices=devices,
            external_references=external_references,
        )

    @classmethod
    def load_pe(
        cls,
        binary: bytes | bytearray | memoryview,
        *,
        maximum_file_size: int,
        **program_options,
    ) -> "BinaryMachineProgram":
        """Consume subject bytes through the existing machine decompiler."""

        program = raise_pe_to_token_multigraph(
            binary, maximum_file_size=maximum_file_size,
        )
        return cls.from_program(program, **program_options)

    def set_speed(self, transitions_per_second: float) -> None:
        self.clock.set_speed(transitions_per_second)

    def set_direction(self, direction: MachineRunDirection) -> None:
        self.runner.set_direction(direction)

    def tick(self, elapsed_seconds: float) -> int:
        """Advance from one external shell tick and publish exactly one flip."""

        return self.runner.regulated_tick(self.clock, elapsed_seconds)

    def pending_external_requests(self, core_index: int = 0) -> tuple[MachineExternalCallRequest, ...]:
        return self.machine.cores[core_index].state.external_requests

    def complete_external_request(
        self,
        request_id: int,
        *,
        result: int = 0,
        core_index: int = 0,
    ):
        """Apply a shell completion and retain it in reversible history."""

        core = self.machine.cores[core_index]
        state = core.state
        matches = tuple(
            request for request in state.external_requests
            if request.request_id == int(request_id)
        )
        if len(matches) != 1:
            raise KeyError(f"external request {request_id} is not pending on core {core_index}")
        completed = complete_external_call_state(
            state, request_id, result=result,
        )
        core.commit_external_completion(completed)
        return completed


__all__ = ["BinaryMachineProgram", "SubjectDeviceBuffers"]
