from dataclasses import replace

from src.compiler.amd64_machine_semantics import PagedByteMemory
from src.compiler.machine_execution import (
    MachineExecutionState, MachineExternalCallCompletion,
    MachineExternalCallRequest, MachineExternalReference,
)
from src.compiler.machine_system_ports import CapabilityGatedExternalPort
from src.compiler.machine_system_tape import MachineSystemTape
from src.compiler.machine_tape_forwarding import (
    TapeForwardingExternalPort, find_recorded_completion,
)
from src.compiler.shell_io import VirtualFileSystemContract, VirtualMount
from src.compiler.virtual_filesystem import VirtualFileEffect, VirtualFileSystemState


def _reference():
    return MachineExternalReference(
        1, 0x1000, "guest-binary", "kernel32.dll", "GetTickCount",
    )


def _request(request_id: int, *, reference=None):
    return MachineExternalCallRequest(
        request_id, reference or _reference(),
        instruction_address=0x2000, return_address=0x2005,
        arguments=(0, 0, 0, 0), stack_pointer=0x3000,
    )


def _recorded_tape():
    """A one-core tape with a real recorded request -> completion transition."""

    memory = PagedByteMemory.empty().map_zeroes(0x4000, 16)
    pending = MachineExecutionState(
        0x2000, memory=memory, external_requests=(_request(5),),
    )
    completed = replace(
        pending, pc=0x2005, external_requests=(),
        registers=(0xDEAD_BEEF,) + (0,) * 15,
        memory=memory.write_unsigned(0x4000, 32, 0x11223344),
    )
    tape = MachineSystemTape(b"MZ subject", 1, checkpoint_interval=8)
    tape.append(0, pending, position=0, event="external_request")
    tape.append(0, completed, position=1, event="external_completion")
    return tape


def test_find_recorded_completion_reconstructs_result_and_memory_writes():
    tape = _recorded_tape()
    new_request = _request(99)  # same reference/arguments, different request id

    completion = find_recorded_completion(tape, new_request)

    assert completion is not None
    assert completion.request_id == 99
    assert completion.result == 0xDEAD_BEEF
    assert len(completion.memory_writes) == 1
    write = completion.memory_writes[0]
    assert write.address == 0x4000
    assert write.data == (0x11223344).to_bytes(4, "little")


def test_find_recorded_completion_is_silent_for_a_different_request_identity():
    tape = _recorded_tape()
    other_reference = MachineExternalReference(
        2, 0x1010, "guest-binary", "kernel32.dll", "ReadFile",
    )
    unmatched = _request(99, reference=other_reference)

    assert find_recorded_completion(tape, unmatched) is None


def test_tape_forwarding_port_falls_back_only_when_the_live_port_cannot_serve():
    tape = _recorded_tape()
    empty_live = CapabilityGatedExternalPort.build({})
    port = TapeForwardingExternalPort(tape=tape, live=empty_live)
    request = _request(99)
    state = MachineExecutionState(0x2000)

    assert port.supports(request)
    completion = port.handle(request, state)
    assert completion is not None
    assert completion.result == 0xDEAD_BEEF

    def _live_handler(req, st):
        return MachineExternalCallCompletion(req.request_id, result=1)

    registered_live = CapabilityGatedExternalPort.build({
        ("kernel32.dll", "GetTickCount"): _live_handler,
    })
    live_port = TapeForwardingExternalPort(tape=tape, live=registered_live)
    live_completion = live_port.handle(request, state)
    assert live_completion.result == 1  # live handler wins over tape forwarding


def test_wider_recorded_effects_are_not_forwarded():
    filesystem = VirtualFileSystemState.create(
        VirtualFileSystemContract(mounts=(VirtualMount.create("/", "memory", access="read_write"),)),
        files={},
    )
    memory = PagedByteMemory.empty()
    pending = MachineExecutionState(
        0x2000, memory=memory, external_requests=(_request(5),),
        virtual_filesystem=filesystem,
    )
    completed = replace(
        pending, pc=0x2005, external_requests=(),
        virtual_filesystem=filesystem.apply(
            VirtualFileEffect("create", "/out.txt", b"hi"),
        ),
    )
    tape = MachineSystemTape(b"MZ subject", 1, checkpoint_interval=8)
    tape.append(0, pending, position=0, event="external_request")
    tape.append(0, completed, position=1, event="external_completion")

    assert find_recorded_completion(tape, _request(99)) is None
