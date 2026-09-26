from dataclasses import replace

import pytest

from src.compiler.amd64_machine_semantics import PagedByteMemory
from src.compiler.machine_execution import (
    MachineExecutionState, MachineExternalCallRequest, MachineExternalReference,
)
from src.compiler.machine_system_tape import MachineSystemTape
from src.compiler.machine_tape_segments import SegmentedMachineTapeStore
from src.compiler.shell_io import VirtualFileSystemContract, VirtualMount
from src.compiler.virtual_filesystem import (
    VirtualFileEffect, VirtualFileSystemState, normalize_virtual_path,
)
from src.compiler.virtual_registry import VirtualRegistryEffect, VirtualRegistryState
from src.compiler.virtual_memory import (
    PAGE_EXECUTE_READWRITE, VirtualMemoryEffect, VirtualMemoryState,
)


def _filesystem():
    return VirtualFileSystemState.create(
        VirtualFileSystemContract(
            current_directory="/work",
            mounts=(VirtualMount.create("/", "memory", access="read_write"),),
        ),
        files={"/work/input.txt": b"hello"},
    )


def test_virtual_paths_are_canonical_and_do_not_escape_root():
    assert normalize_virtual_path(r"C:\tools\..\subject.exe") == "/c/subject.exe"
    assert normalize_virtual_path("../../output.txt", "/work/jobs") == "/output.txt"


def test_virtual_filesystem_effects_are_immutable_and_reversible_values():
    before = _filesystem()
    after = before.apply(VirtualFileEffect("write", "input.txt", b"!", offset=5))
    after = after.apply(VirtualFileEffect("create", "output.txt", b"result"))

    assert before.read("input.txt") == b"hello"
    assert after.read("input.txt") == b"hello!"
    assert after.read("output.txt") == b"result"
    assert before.generation == 0
    assert after.generation == 2


def test_read_only_mount_rejects_mutation():
    filesystem = VirtualFileSystemState.create(
        VirtualFileSystemContract(mounts=(
            VirtualMount.create("/", "memory", access="read_write"),
            VirtualMount.create("/bundle", "bundle", source="demo"),
        )),
        files={"/bundle/program.exe": b"MZ"},
    )
    with pytest.raises(PermissionError, match="read-only"):
        filesystem.apply(VirtualFileEffect("write", "/bundle/program.exe", b"X"))


def test_directory_handles_and_cursors_are_immutable_tape_state():
    filesystem = _filesystem()
    opened = filesystem.apply(VirtualFileEffect(
        "open", "/work/*.txt", handle=filesystem.next_handle,
        entries=("/work/input.txt",),
    ))
    advanced = opened.apply(VirtualFileEffect(
        "advance", "/work/*.txt", handle=filesystem.next_handle,
    ))
    closed = advanced.apply(VirtualFileEffect(
        "close", "/", handle=filesystem.next_handle,
    ))
    assert filesystem.handles == {}
    assert opened.handles[0x1000].position == 0
    assert advanced.handles[0x1000].position == 1
    assert closed.handles == {}


def test_system_tape_round_trips_checkpoints_memory_vfs_and_rewind(tmp_path):
    filesystem = _filesystem()
    memory = PagedByteMemory.empty().map_zeroes(0x1000, 4096)
    first = MachineExecutionState(
        0x1000, memory=memory, virtual_filesystem=filesystem,
        environment_state={"PATH": "/tools"},
        text_state={"msvcrt.locale.ctype": "C"},
    )
    second = replace(
        first, pc=0x1001, steps=1,
        memory=memory.write_unsigned(0x1010, 64, 0x1234),
        virtual_filesystem=filesystem.apply(
            VirtualFileEffect("write", "/work/input.txt", b"!", offset=5),
        ),
    )
    rewound = first
    tape = MachineSystemTape(b"MZ subject", 1, checkpoint_interval=8)
    dynamic_reference = MachineExternalReference(
        7, 0xFFFF800000000060, "guest-binary", "kernel32.dll", "Example",
    )
    tape.catalog_external_reference(dynamic_reference)
    tape.append(0, first, position=0, event="load")
    tape.append(0, second, position=1, event="forward")
    tape.append(0, rewound, position=0, event="backward")
    warning = tape.annotate(
        "uncertain_semantics", "this might be wrong",
        color="#ff8800", severity="suspect", sequence=1, core=0,
        position=1, address=0x1001, metadata={"reviewer": "human"},
    )

    assert tape.resume_state(sequence=1).memory.read_unsigned(0x1010, 64) == 0x1234
    assert tape.resume_state(sequence=1).virtual_filesystem.read("input.txt") == b"hello!"
    assert tape.resume_state().pc == first.pc
    assert tape.resume_state().virtual_filesystem.read("input.txt") == b"hello"
    assert tape.annotations_at(1, core=0) == (warning,)
    assert tape.annotations_at(0, core=0) == ()

    path = tape.write(tmp_path / "machine.tape.jsonl")
    loaded = MachineSystemTape.read(path)
    assert loaded.subject_binary == b"MZ subject"
    assert loaded.external_references == [dynamic_reference]
    assert loaded.resume_state().memory.read_unsigned(0x1010, 64) == 0
    assert loaded.resume_state().environment_state == {"PATH": "/tools"}
    assert loaded.resume_state().text_state == {"msvcrt.locale.ctype": "C"}
    assert loaded.annotations[0].message == "this might be wrong"
    assert loaded.annotations[0].color == "#ff8800"
    assert loaded.annotations[0].metadata == (("reviewer", "human"),)


def test_open_file_cursor_metadata_and_attributes_survive_tape_round_trip(tmp_path):
    filesystem = _filesystem()
    filesystem = filesystem.apply(VirtualFileEffect(
        "open", "/work/input.txt", handle=filesystem.next_handle,
        mode="file:c0000000:3",
    ))
    filesystem = filesystem.apply(VirtualFileEffect(
        "seek", "/work/input.txt", handle=0x1000, offset=4,
    ))
    filesystem = filesystem.apply(VirtualFileEffect(
        "set_times", "/work/input.txt", created_time=11,
        accessed_time=22, modified_time=33,
    ))
    filesystem = filesystem.apply(VirtualFileEffect(
        "set_attributes", "/work/input.txt", attributes=0x22,
    ))
    state = MachineExecutionState(pc=0x1000, virtual_filesystem=filesystem)
    tape = MachineSystemTape(b"MZ", 1)
    tape.append(0, state, position=0, event="file-state")
    loaded = MachineSystemTape.read(tape.write(tmp_path / "file-state.jsonl"))
    restored = loaded.resume_state().virtual_filesystem
    assert restored.handles[0x1000].mode == "file:c0000000:3"
    assert restored.handles[0x1000].position == 4
    assert restored.stat("/work/input.txt").created_time == 11
    assert restored.stat("/work/input.txt").accessed_time == 22
    assert restored.stat("/work/input.txt").modified_time == 33
    assert restored.stat("/work/input.txt").attributes == 0x22


def test_virtual_registry_handles_values_and_metadata_round_trip_on_tape(tmp_path):
    registry = VirtualRegistryState.create()
    registry = registry.apply(VirtualRegistryEffect(
        "create_key", "hkey_current_user\\Software\\Turing",
    ))
    registry = registry.apply(VirtualRegistryEffect(
        "open", "hkey_current_user\\software\\turing",
        handle=registry.next_handle, access=0xF003F,
    ))
    registry = registry.apply(VirtualRegistryEffect(
        "set_value", "hkey_current_user\\software\\turing",
        value_name="Mode", value_type=1, data="exact\0".encode("utf-16le"),
    ))
    state = MachineExecutionState(pc=0x1000, virtual_registry=registry)
    tape = MachineSystemTape(b"MZ", 1)
    tape.append(0, state, position=0, event="registry-state")
    loaded = MachineSystemTape.read(tape.write(tmp_path / "registry-state.jsonl"))
    restored = loaded.resume_state().virtual_registry
    assert restored.handles[0x4000].access == 0xF003F
    value = restored.keys[
        "hkey_current_user\\software\\turing"
    ].values["mode"]
    assert value.name == "Mode"
    assert value.value_type == 1
    assert value.data == "exact\0".encode("utf-16le")


def test_virtual_pipe_handles_and_buffer_round_trip_on_segmented_tape(tmp_path):
    state = MachineExecutionState(
        pc=0x1000,
        system_state={
            "windows.pipe.1.capacity": 4096,
            "windows.pipe.1.readers": 1,
            "windows.pipe.1.writers": 2,
            "windows.handle.256.kind": 4,
            "windows.handle.256.id": 1,
            "windows.handle.256.end": 1,
        },
        device_state={"pipe.1": b"pending bytes"},
    )
    tape = MachineSystemTape(b"MZ", 1)
    tape.append(0, state, position=0, event="pipe-state")
    source = tape.write(tmp_path / "pipe.jsonl")
    SegmentedMachineTapeStore.import_jsonl(
        source, tmp_path / "pipe.segmented-tape", records_per_segment=1,
    )
    loaded = SegmentedMachineTapeStore(tmp_path / "pipe.segmented-tape")
    restored = loaded.resume_state()
    assert restored.system_state["windows.pipe.1.writers"] == 2
    assert restored.system_state["windows.handle.256.end"] == 1
    assert restored.device_state["pipe.1"] == b"pending bytes"


def test_virtual_memory_regions_and_committed_pages_round_trip_on_tape(tmp_path):
    virtual_memory = VirtualMemoryState.create().apply(VirtualMemoryEffect(
        "allocate", 0x10000000000, 8192, PAGE_EXECUTE_READWRITE,
    ))
    memory = PagedByteMemory.empty().map_zeroes(0x10000000000, 8192)
    memory = memory.map_bytes(0x10000000000, b"\x90\xc3")
    state = MachineExecutionState(
        pc=0x10000000000, memory=memory, virtual_memory=virtual_memory,
    )
    tape = MachineSystemTape(b"MZ", 1)
    tape.append(0, state, position=0, event="virtual-memory-state")
    loaded = MachineSystemTape.read(tape.write(tmp_path / "virtual-memory.jsonl"))
    restored = loaded.resume_state()
    assert restored.virtual_memory.is_executable(0x10000000000)
    assert restored.virtual_memory.regions[0x10000000000].managed
    assert bytes(restored.memory[0x10000000000 + i] for i in range(2)) == b"\x90\xc3"


def test_tape_annotations_support_ranges_colors_and_superseding_notes():
    state = MachineExecutionState(
        0x1000, memory=PagedByteMemory.empty().map_zeroes(0x1000, 4096),
    )
    tape = MachineSystemTape(b"MZ", 1)
    tape.append(0, state, position=0, event="load")
    tape.append(0, replace(state, pc=0x1001), position=1, event="forward")
    first = tape.annotate(
        "decoder_confidence", "review this span", color="amber",
        severity="caution", sequence=0, end_sequence=1,
    )
    replacement = tape.annotate(
        "decoder_confidence", "verified after review", color="green",
        severity="verified", sequence=1, supersedes=first.annotation_id,
    )
    assert tape.annotations_at(0) == (first,)
    assert tape.annotations_at(1) == (first, replacement)
    with pytest.raises(ValueError, match="annotation color"):
        tape.annotate("bad", "bad color", color="orange-ish")


def test_tape_dependency_graph_links_state_parents_and_external_completion(tmp_path):
    memory = PagedByteMemory.empty().map_zeroes(0x1000, 4096)
    initial = MachineExecutionState(0x1000, memory=memory)
    reference = MachineExternalReference(
        1, 0xFFFF800000000000, "guest-binary", "kernel32.dll", "Example",
    )
    request = MachineExternalCallRequest(
        7, reference, 0x1000, 0x1005, (1, 2, 3, 4), 0x1FF8,
    )
    waiting = replace(initial, pc=reference.target_address, external_requests=(request,))
    completed = replace(
        waiting, pc=0x1005, external_requests=(), halted=True, exit_code=7,
    )
    tape = MachineSystemTape(b"MZ", 1)
    tape.append(0, initial, position=0, event="load")
    tape.append(0, waiting, position=1, event="forward")
    tape.append(0, completed, position=2, event="external_completion")

    graph = tape.dependency_graph()
    assert graph.lineage(2) == (0, 1, 2)
    assert graph.nodes[1].parent_sequence == 0
    assert graph.nodes[2].dependencies == (("external_request", 1),)
    loaded = MachineSystemTape.read(tape.write(tmp_path / "graph.tape.jsonl"))
    assert loaded.dependency_graph().nodes == graph.nodes
    assert loaded.resume_state().pc == 0x1005
    assert loaded.resume_state().halted and loaded.resume_state().exit_code == 7
