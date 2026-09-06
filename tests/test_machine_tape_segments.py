from dataclasses import replace

from src.compiler.machine_execution import MachineExecutionState, MachineExternalReference
from src.compiler.machine_system_tape import MachineSystemTape
from src.compiler.machine_tape_segments import SegmentedMachineTapeStore


def test_json_tape_streams_into_content_addressed_memory_bounded_segments(tmp_path):
    tape = MachineSystemTape(b"binary-subject", 1, checkpoint_interval=2)
    state = MachineExecutionState(pc=0x1000)
    for position in range(7):
        active = replace(state, pc=0x1000 + position, steps=position)
        tape.append(0, active, position=position, event="load" if position == 0 else "forward")
    tape.catalog_external_reference(MachineExternalReference(
        1, 0xFFFF800000000000, "guest-binary", "kernel32.dll", "WriteConsoleW",
    ))
    tape.annotate("moment", "bounded segment proof", sequence=4)
    tape.append(
        0, replace(state, pc=0x3000, steps=7),
        position=7, event="runtime_dispatch",
        metadata={"targets": [0x3000, 0x3010]},
    )
    source = tape.write(tmp_path / "source.tape.jsonl")

    store = SegmentedMachineTapeStore.import_jsonl(
        source, tmp_path / "segments", records_per_segment=2,
    )

    assert store.subject_binary == b"binary-subject"
    assert store.record_count == 8
    assert len(store.segments) == 4
    assert all(segment.record_count <= 2 for segment in store.segments)
    assert store.lineage() == tuple(range(8))
    assert store.resume_state().pc == 0x3000
    assert store.resume_state(sequence=3).steps == 3
    assert len(store._cached_records) <= 2
    assert store.external_references[0].symbol == "WriteConsoleW"
    assert store.annotations[0].message == "bounded segment proof"
    assert store.runtime_dispatch_targets == (0x3000, 0x3010)

    resumed = store.resume_state()
    store.append(
        0, replace(resumed, pc=0x2000, steps=8),
        position=8, event="forward",
    )
    store.annotate("continued", "appended without materializing old segments")
    store.flush()
    reopened = SegmentedMachineTapeStore(store.root)
    assert reopened.record_count == 9
    assert reopened.resume_state().pc == 0x2000
    assert reopened.annotations[-1].feature == "continued"
    assert reopened.runtime_dispatch_targets == (0x3000, 0x3010)
    assert len(reopened._cached_records) <= 2


def test_segment_store_rejects_tampered_content_before_resume(tmp_path):
    tape = MachineSystemTape(b"subject", 1)
    tape.append(0, MachineExecutionState(pc=1), position=0, event="load")
    source = tape.write(tmp_path / "source.tape.jsonl")
    store = SegmentedMachineTapeStore.import_jsonl(source, tmp_path / "store")
    segment = store.segments[0]
    path = store.root / "segments" / f"{segment.digest}.json.gz"
    encoded = bytearray(path.read_bytes())
    encoded[-1] ^= 1
    path.write_bytes(encoded)

    reloaded = SegmentedMachineTapeStore(store.root)
    try:
        reloaded.resume_state()
    except (OSError, ValueError):
        pass
    else:
        raise AssertionError("tampered tape segment was accepted")


def test_graph_crop_is_independent_position_zero_state_with_origin_receipt(tmp_path):
    tape = MachineSystemTape(b"binary-subject", 1, checkpoint_interval=2)
    for position in range(6):
        tape.append(
            0, MachineExecutionState(
                pc=0x1000 + position, steps=position,
                system_state={"machine.memory.page_size": 4096},
            ),
            position=position, event="load" if position == 0 else "forward",
        )
    source = tape.write(tmp_path / "source.tape.jsonl")
    store = SegmentedMachineTapeStore.import_jsonl(
        source, tmp_path / "source.segmented", records_per_segment=2,
    )

    crop = store.crop(tmp_path / "crop.segmented", sequence=3)

    assert crop.record_count == 1
    assert crop.record(0)["event"] == "graph_crop_root"
    assert crop.record(0)["position"] == 0
    assert crop.record(0)["parent_sequence"] is None
    assert crop.resume_state().pc == 0x1003
    assert crop.resume_state().steps == 3
    assert crop.origin_receipt["schema"] == "turing-machine-graph-crop-origin-v1"
    assert crop.origin_receipt["source_sequences"] == [3]
    assert len(crop.origin_receipt["state_digest"]) == 64

    # The source can disappear and the cropped machine remains a complete root.
    for path in sorted(store.root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    store.root.rmdir()
    reopened = SegmentedMachineTapeStore(crop.root)
    assert reopened.resume_state() == crop.resume_state()
    reopened.begin_append()
    reopened.append(
        0, replace(reopened.resume_state(), pc=0x2000, steps=4),
        position=1, event="forward",
    )
    reopened.flush()
    assert SegmentedMachineTapeStore(crop.root).resume_state().pc == 0x2000
