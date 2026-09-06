from dataclasses import replace

from src.compiler.machine_execution import MachineExecutionState
from src.compiler.machine_path_segments import SegmentedMachinePathStateStore


def _state(position: int, value: int) -> MachineExecutionState:
    return MachineExecutionState(
        pc=0x400000 + position,
        registers=(value, *(0 for _ in range(15))),
        steps=position,
    )


def test_exact_path_heads_share_parent_prefix_and_stream_reopenable_suffixes(tmp_path):
    store = SegmentedMachinePathStateStore(
        tmp_path / "path-states", create=True, states_per_segment=2,
    )
    store.create_head("root", 0, _state(0, 0))
    for position in range(1, 5):
        store.append_state("root", position, _state(position, position))
    store.create_head(
        "left", 2, _state(2, 2), parent_head_id="root", fork_position=2,
        constraints=({"expression": "input == left"},),
    )
    store.create_head(
        "right", 2, _state(2, 2), parent_head_id="root", fork_position=2,
    )
    for identity in ("left", "right"):
        store.append_state(identity, 3, _state(3, 30))
        store.append_state(identity, 4, _state(4, 40))
    store.flush()

    reopened = SegmentedMachinePathStateStore(store.root)
    assert [position for position, _state_value in reopened.iter_states("left")] == [0, 1, 2, 3, 4]
    assert reopened.latest_state("left")[1].registers[0] == 40
    assert reopened.heads["left"].constraints[0]["expression"] == "input == left"
    assert reopened.heads["left"].segments[0].digest == reopened.heads["right"].segments[0].digest
    assert reopened.heads["left"].segments[1].digest == reopened.heads["right"].segments[1].digest
    assert reopened.cached_state_count <= 2

    reopened.append_state("left", 5, _state(5, 50))
    assert reopened.resident_head_ids == ("left",)
    reopened.release_head("left")
    assert reopened.resident_head_ids == ()
    reopened.clear_read_cache()
    assert reopened.cached_state_count == 0

    # A released path lazily reloads only its tip and can continue into a new
    # immutable segment without hydrating its complete shared ancestry.
    reopened.append_state("left", 6, _state(6, 60))
    reopened.release_all()
    assert reopened.resident_head_ids == ()
    appended = SegmentedMachinePathStateStore(store.root)
    assert appended.latest_state("left")[1].registers[0] == 60
    assert appended.cached_state_count <= 2


def test_path_state_digest_tampering_fails_closed(tmp_path):
    store = SegmentedMachinePathStateStore(tmp_path / "path-states", create=True)
    head = store.create_head("root", 0, _state(0, 0))
    segment = head.segments[0]
    path = store.root / "objects" / f"{segment.digest}.json.gz"
    path.write_bytes(b"not the retained object")

    reopened = SegmentedMachinePathStateStore(store.root)
    try:
        tuple(reopened.iter_states("root"))
    except (OSError, ValueError) as error:
        assert "digest" in str(error).casefold() or isinstance(error, OSError)
    else:  # pragma: no cover - corruption must never be accepted
        raise AssertionError("corrupt path state segment was accepted")
