from dataclasses import replace
from types import SimpleNamespace

from src.compiler.machine_execution import (
    MachineExecutionOrchestrator,
    ReversibleMachineExecutor,
)
from src.compiler.machine_path_forest import (
    MachinePathBranch,
    MachinePathConstraint,
    MachinePathForest,
    MachinePathHeadStatus,
)
from src.compiler.machine_path_segments import SegmentedMachinePathStateStore
from src.compiler.machine_trace_ssa import (
    lift_path_state_head_to_trace_ssa,
    segment_path_state_head_to_trace_ssa,
)
from src.compiler.machine_trace_ssa_segments import SegmentedMachineTraceSSAStore
from src.compiler.machine_reference_vocabulary import (
    MachineSemanticToken,
    X86InstructionToken,
)


def _executor():
    instructions = tuple(SimpleNamespace(
        address=0x401000 + offset,
        encoded=b"\x90",
        token=X86InstructionToken.NOP,
        semantic=MachineSemanticToken.INTEGER_ADD,
        operands=(),
    ) for offset in range(8))
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x400000, entrypoint_rva=0x1000),
        functions=(SimpleNamespace(report=SimpleNamespace(instructions=instructions)),),
    )

    def increment(state, instruction):
        registers = list(state.registers)
        registers[0] += 1
        return replace(
            state,
            pc=instruction.address + 1,
            registers=tuple(registers),
            steps=state.steps + 1,
        )

    orchestrator = MachineExecutionOrchestrator(
        program,
        effect_handlers={int(MachineSemanticToken.INTEGER_ADD): increment},
    )
    return ReversibleMachineExecutor.create(orchestrator)


def _rax(value):
    def transform(state):
        registers = list(state.registers)
        registers[0] = value
        return replace(state, registers=tuple(registers))
    return transform


def test_read_heads_fork_from_any_history_position_and_keep_sibling_worlds():
    executor = _executor()
    executor.step_forward()
    executor.step_forward()
    forest = MachinePathForest(executor, tape_sequence=12)

    left, right = forest.fork_read_head(
        0,
        (
            MachinePathBranch(
                "input-a",
                (MachinePathConstraint("terminal", "line == 'a'", 12),),
                _rax(10),
            ),
            MachinePathBranch(
                "input-b",
                (MachinePathConstraint("terminal", "line == 'b'", 12),),
                _rax(20),
            ),
        ),
        history_position=1,
    )

    assert forest.heads[0].status is MachinePathHeadStatus.FORKED
    assert forest.children(0) == (left.head_id, right.head_id)
    assert left.fork_history_position == right.fork_history_position == 1
    assert left.fork_tape_sequence == right.fork_tape_sequence == 12
    assert left.state.registers[0] == 10
    assert right.state.registers[0] == 20
    assert executor.state.registers[0] == 2
    graph = forest.provenance_graph()
    assert graph["world_axis"] == "forked-read-head"
    assert graph["thread_axis"] == "independent-not-represented-here"
    assert graph["edges"] == (
        {
            "source": 0, "target": 1, "kind": "fork_read_head",
            "history_position": 1, "tape_sequence": 12,
        },
        {
            "source": 0, "target": 2, "kind": "fork_read_head",
            "history_position": 1, "tape_sequence": 12,
        },
    )


def test_possible_world_heads_advance_in_parallel_without_shared_writes():
    forest = MachinePathForest(_executor())
    left, right = forest.fork_read_head(
        0,
        (
            MachinePathBranch("left", state_transform=_rax(100)),
            MachinePathBranch("right", state_transform=_rax(200)),
        ),
    )

    advances = forest.advance_parallel(
        (left.head_id, right.head_id), maximum_steps=3, maximum_workers=2,
    )

    assert [item.transitions for item in advances] == [3, 3]
    assert left.state.registers[0] == 103
    assert right.state.registers[0] == 203
    assert forest.heads[0].state.registers[0] == 0


def test_parallel_worlds_stream_exact_suffixes_and_reverse_from_cold_segments(tmp_path):
    store = SegmentedMachinePathStateStore(
        tmp_path / "world-states", create=True, states_per_segment=2,
    )
    forest = MachinePathForest(_executor(), exact_state_store=store)
    left, right = forest.fork_read_head(
        0,
        (
            MachinePathBranch("left", state_transform=_rax(100)),
            MachinePathBranch("right", state_transform=_rax(200)),
        ),
    )

    forest.advance_parallel(
        (left.head_id, right.head_id), maximum_steps=3, maximum_workers=2,
    )
    reopened = SegmentedMachinePathStateStore(store.root)

    assert [position for position, _state in reopened.iter_states(left.head_id)] == [0, 1, 2, 3, 4]
    assert reopened.latest_state(left.head_id)[1].registers[0] == 103
    assert reopened.latest_state(right.head_id)[1].registers[0] == 203
    assert forest.provenance_graph()["nodes"][1]["exact_state_segments"] >= 2

    for _ in range(4):
        left.executor.step_backward()
    assert left.executor.position == 0
    assert left.executor.state.registers[0] == 0
    assert len(left.executor._states) <= store.states_per_segment

    trace = lift_path_state_head_to_trace_ssa(reopened, left.head_id)
    assert trace.specialization == "observed-possible-world-path"
    assert trace.operations[-1].address == 0x401002
    assert trace.operations[-1].instruction_token == "NOP"
    assert trace.operations[-1].semantic_token == "INTEGER_ADD"
    assert trace.operations[-1].tape_dependencies == (("path_parent", 3),)
    assert trace.final_values["register.rax"] == "register.rax@4"

    trace_store = SegmentedMachineTraceSSAStore(tmp_path / "world-ssa", create=True)
    segment_path_state_head_to_trace_ssa(
        reopened, left.head_id, trace_store, operations_per_segment=2,
    )
    reopened_trace = SegmentedMachineTraceSSAStore(trace_store.root)
    assert reopened_trace.operation_count(left.head_id) == 4
    assert reopened_trace.heads[str(left.head_id)].final_values["register.rax"] == "register.rax@4"
    assert reopened_trace.cached_operation_count <= 2
    assert reopened.cached_state_count <= store.states_per_segment
