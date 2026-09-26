from dataclasses import replace

from src.compiler.machine_execution import MachineExecutionState
from src.compiler.machine_system_tape import MachineSystemTape
from src.compiler.machine_trace_ssa import lift_tape_lineage_to_trace_ssa
from src.compiler.machine_trace_ssa_segments import SegmentedMachineTraceSSAStore


def _trace(register_values):
    tape = MachineSystemTape(b"subject", 1)
    state = MachineExecutionState(pc=0x100)
    tape.append(0, state, position=0, event="load")
    for position, value in enumerate(register_values, 1):
        state = replace(
            state, pc=0x100 + position,
            registers=(value, *(0 for _ in range(15))), steps=position,
        )
        tape.append(0, state, position=position, event="forward")
    return lift_tape_lineage_to_trace_ssa(tape)


def test_branching_ssa_heads_share_parent_prefix_and_stream_bounded_chunks(tmp_path):
    root = _trace((1, 2, 3, 4))
    left = _trace((1, 2, 30, 40, 50))
    right = _trace((1, 2, 300, 400))
    store = SegmentedMachineTraceSSAStore(tmp_path / "ssa-segments", create=True)
    store.add_head("root", root, operations_per_segment=2)
    store.add_head(
        "left", left, parent_head_id="root", fork_sequence=2,
        constraints=({"expression": "input == left"},), operations_per_segment=2,
    )
    store.add_head(
        "right", right, parent_head_id="root", fork_sequence=2,
        constraints=({"expression": "input == right"},), operations_per_segment=2,
    )

    reopened = SegmentedMachineTraceSSAStore(store.root)
    assert [item["sequence"] for item in reopened.iter_operations("left")] == [1, 2, 3, 4, 5]
    assert [item["sequence"] for item in reopened.iter_operations("right")] == [1, 2, 3, 4]
    assert sum(item.operation_count for item in reopened.heads["left"].segments) == 3
    assert sum(item.operation_count for item in reopened.heads["right"].segments) == 2
    assert reopened.cached_operation_count <= 2
    assert reopened.heads["left"].constraints[0]["expression"] == "input == left"


def test_identical_ssa_suffix_chunks_are_content_addressed_once(tmp_path):
    trace = _trace((1, 2, 3, 4))
    store = SegmentedMachineTraceSSAStore(tmp_path / "ssa-segments", create=True)
    first = store.add_head("first", trace, operations_per_segment=2)
    second = store.add_head("second", trace, operations_per_segment=2)

    assert [item.digest for item in first.segments] == [item.digest for item in second.segments]
    assert len(tuple((store.root / "objects").glob("*.json.gz"))) == 2


def test_ssa_operation_generator_is_consumed_into_bounded_segments(tmp_path):
    trace = _trace(tuple(range(1, 10)))
    observed_residency = []
    store = SegmentedMachineTraceSSAStore(tmp_path / "ssa-segments", create=True)

    def operations():
        for operation in trace.operations:
            observed_residency.append(store.cached_operation_count)
            yield operation.to_mapping()

    head = store.add_operation_stream(
        "stream", operations(), core=trace.core,
        specialization=trace.specialization, final_values=trace.final_values,
        operations_per_segment=3,
    )

    assert sum(item.operation_count for item in head.segments) == 9
    assert all(item.operation_count <= 3 for item in head.segments)
    assert observed_residency == [0] * 9
    assert store.operation_count("stream") == 9
    assert store.cached_operation_count <= 3
    store.clear_read_cache()
    assert store.cached_operation_count == 0
