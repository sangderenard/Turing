from dataclasses import replace
from types import SimpleNamespace

from src.compiler.amd64_machine_semantics import PagedByteMemory
from src.compiler.machine_execution import MachineExecutionState
from src.compiler.machine_system_tape import MachineSystemTape
from src.compiler.machine_trace_ssa import lift_tape_lineage_to_trace_ssa
from src.compiler.virtual_registry import VirtualRegistryEffect, VirtualRegistryState
from src.compiler.virtual_memory import PAGE_READWRITE, VirtualMemoryEffect, VirtualMemoryState


def _instruction(address: int):
    return SimpleNamespace(
        address=address,
        token=SimpleNamespace(name="ADD_RM64_R64"),
        semantic=SimpleNamespace(name="INTEGER_ADD"),
        encoded=b"\x48\x01\xd8",
    )


def test_tape_lineage_becomes_versioned_effect_aware_trace_ssa():
    memory = PagedByteMemory.empty().map_zeroes(0x1000, 0x1000)
    initial = MachineExecutionState(pc=0x100, memory=memory)
    arithmetic = replace(
        initial, pc=0x103, registers=(2, *(0 for _ in range(15))), steps=1,
    )
    output = replace(
        arithmetic,
        device_state={"console.output": b"hello\r\n"},
        device_generations={"console.output": 1},
    )
    finished = replace(
        output, pc=0x106, registers=(3, *(0 for _ in range(15))), steps=2,
    )
    tape = MachineSystemTape(b"subject", 1)
    tape.append(0, initial, position=0, event="load")
    tape.append(0, arithmetic, position=1, event="forward")
    tape.append(0, output, position=2, event="external_completion")
    tape.append(0, finished, position=3, event="forward")

    trace = lift_tape_lineage_to_trace_ssa(
        tape, instruction_lookup=lambda address: _instruction(address),
    )

    assert len(trace.operations) == 3
    assert trace.operations[0].instruction_token == "ADD_RM64_R64"
    assert trace.operations[0].semantic_token == "INTEGER_ADD"
    assert trace.operations[0].pure is True
    assert trace.operations[1].effect_domains == ("control", "device")
    assert trace.final_values["device.console.output"].startswith(
        "device.console.output@"
    )
    assert trace.to_mapping()["specialization"] == "observed-tape-lineage"


def test_trace_ssa_can_slice_terminal_math_from_machine_bookkeeping():
    initial = MachineExecutionState(pc=0x100)
    first = replace(initial, pc=0x101, registers=(4, *(0 for _ in range(15))), steps=1)
    output = replace(first, device_state={"console.output": b"four"})
    unrelated = replace(output, pc=0x102, registers=(9, *(0 for _ in range(15))), steps=2)
    tape = MachineSystemTape(b"subject", 1)
    for position, (event, state) in enumerate((
        ("load", initial), ("forward", first),
        ("external_completion", output), ("forward", unrelated),
    )):
        tape.append(0, state, position=position, event=event)
    trace = lift_tape_lineage_to_trace_ssa(tape)

    data_slice = trace.backward_slice(
        ("device.console.output",), include_control=False,
    )

    assert [item.event for item in data_slice.operations] == ["external_completion"]
    witness = data_slice.to_mapping()["reduction_witness"]
    assert witness["rewrite"] == "backward-slice"
    assert witness["seed_resources"] == ("device.console.output",)
    assert witness["retained_source_sequences"] == (2,)
    assert witness["removed_source_sequences"] == (1, 3)
    assert trace.reduction_summary(data_slice) == {
        "source_operations": 3,
        "retained_operations": 1,
        "removed_operations": 2,
        "retained_pure_operations": 0,
        "retained_effect_operations": 1,
    }


def test_trace_ssa_catalogues_registry_as_a_distinct_effect_domain():
    registry = VirtualRegistryState.create()
    initial = MachineExecutionState(pc=0x100, virtual_registry=registry)
    changed = replace(
        initial,
        virtual_registry=registry.apply(VirtualRegistryEffect(
            "create_key", "hkey_current_user\\Software\\Turing",
        )),
    )
    tape = MachineSystemTape(b"subject", 1)
    tape.append(0, initial, position=0, event="load")
    tape.append(0, changed, position=1, event="external_completion")
    trace = lift_tape_lineage_to_trace_ssa(tape)
    assert trace.operations[0].effect_domains == ("control", "registry")
    assert "registry.state" in trace.final_values


def test_trace_ssa_catalogues_virtual_memory_mapping_effects():
    virtual_memory = VirtualMemoryState.create()
    initial = MachineExecutionState(pc=0x100, virtual_memory=virtual_memory)
    changed = replace(
        initial,
        virtual_memory=virtual_memory.apply(VirtualMemoryEffect(
            "allocate", 0x10000000000, 4096, PAGE_READWRITE,
        )),
    )
    tape = MachineSystemTape(b"subject", 1)
    tape.append(0, initial, position=0, event="load")
    tape.append(0, changed, position=1, event="external_completion")
    trace = lift_tape_lineage_to_trace_ssa(tape)
    assert trace.operations[0].effect_domains == ("control", "virtual_memory")
    assert "virtual_memory.state" in trace.final_values


def test_trace_ssa_catalogues_pipe_transport_as_a_distinct_effect_domain():
    initial = MachineExecutionState(
        pc=0x100,
        system_state={"windows.pipe.1.writers": 1},
        device_state={"pipe.1": b""},
    )
    changed = replace(
        initial,
        device_state={"pipe.1": b"payload"},
        device_generations={"pipe.1": 1},
    )
    tape = MachineSystemTape(b"subject", 1)
    tape.append(0, initial, position=0, event="load")
    tape.append(0, changed, position=1, event="external_completion")

    trace = lift_tape_lineage_to_trace_ssa(tape)

    assert trace.operations[0].effect_domains == ("control", "device", "pipe")
    assert "device.pipe.1" in trace.final_values
