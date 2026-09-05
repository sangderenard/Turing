from types import SimpleNamespace

from src.compiler.amd64_machine_semantics import default_effect_handlers
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator, MachineExecutionState, ReversibleMachineExecutor,
)
from src.compiler.machine_fork_exploration import (
    ForkExplorationPolicy, explore_forking_paths,
)
from src.compiler.machine_path_forest import MachinePathForest, MachinePathHeadStatus
from src.compiler.machine_reference_vocabulary import (
    MachineSemanticToken, RelativeAddressOperand,
)


def _conditional_jump(address: int, target: int) -> SimpleNamespace:
    return SimpleNamespace(
        address=address, encoded=b"\x75\x00",
        semantic=MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        token=SimpleNamespace(name="JNE_REL8"),
        operands=(
            RelativeAddressOperand(displacement=0, width=8, target_address=target),
        ),
    )


def _forest(*instructions: SimpleNamespace, entry: int) -> MachinePathForest:
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=entry, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(instructions=instructions)),),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )
    reversible = ReversibleMachineExecutor.create(
        executor, MachineExecutionState(pc=entry),
    )
    return MachinePathForest(reversible, maximum_heads=64)


def test_auto_fork_explores_both_sides_of_a_single_conditional_branch():
    entry = 0x2000
    target, fallthrough = 0x2100, entry + 2
    forest = _forest(_conditional_jump(entry, target), entry=entry)

    terminal = explore_forking_paths(
        forest, 0,
        policy=ForkExplorationPolicy(
            dfs_bias=0.5, maximum_forks=8, maximum_transitions_per_head=4,
        ),
    )

    pcs = sorted(head.executor.state.pc for head in terminal)
    assert pcs == sorted((target, fallthrough))
    assert all(head.status is MachinePathHeadStatus.BLOCKED for head in terminal)
    # Neither branch executed as a live guest instruction (nothing decoded
    # at either destination) -- both PCs came from fork_read_head's
    # state_transform setting pc directly, matching what the ordinary
    # CONDITIONAL_RELATIVE_JUMP handler in _step_decoded would have set pc
    # to for whichever single answer predicate_handler would have given.
    for head in terminal:
        assert head.parent_head_id == 0
        assert head.branch_label in {"taken", "not-taken"}


def _two_level_tree():
    # entry -> {A-taken -> {AA-taken, AA-not-taken}, A-not-taken -> {AN-taken, AN-not-taken}}
    entry = 0x2000
    a_taken, a_not_taken = 0x2100, entry + 2
    aa_taken, aa_not_taken = 0x2120, a_taken + 2
    an_taken, an_not_taken = 0x2110, a_not_taken + 2
    forest = _forest(
        _conditional_jump(entry, a_taken),
        _conditional_jump(a_taken, aa_taken),
        _conditional_jump(a_not_taken, an_taken),
        entry=entry,
    )
    leaves = {aa_taken, aa_not_taken, an_taken, an_not_taken}
    return forest, leaves


def test_dfs_and_bfs_biases_visit_the_same_leaves_in_a_different_order():
    forest_dfs, leaves = _two_level_tree()
    dfs_terminal = explore_forking_paths(
        forest_dfs, 0,
        policy=ForkExplorationPolicy(
            dfs_bias=1.0, maximum_forks=16, maximum_transitions_per_head=4,
        ),
    )
    forest_bfs, _leaves = _two_level_tree()
    bfs_terminal = explore_forking_paths(
        forest_bfs, 0,
        policy=ForkExplorationPolicy(
            dfs_bias=0.0, maximum_forks=16, maximum_transitions_per_head=4,
        ),
    )

    dfs_pcs = [head.executor.state.pc for head in dfs_terminal]
    bfs_pcs = [head.executor.state.pc for head in bfs_terminal]

    assert set(dfs_pcs) == leaves
    assert set(bfs_pcs) == leaves
    assert len(dfs_pcs) == len(bfs_pcs) == 4
    # Pure BFS visits every head produced by the first fork level before
    # descending into either of their children, so its terminal head ids
    # come out in strictly increasing order; pure DFS drives the
    # most-recently-forked head to a leaf before ever returning to its
    # sibling, so it does not.
    bfs_ids = [head.head_id for head in bfs_terminal]
    dfs_ids = [head.head_id for head in dfs_terminal]
    assert bfs_ids == sorted(bfs_ids)
    assert dfs_ids != sorted(dfs_ids)
