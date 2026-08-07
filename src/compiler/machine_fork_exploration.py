"""Auto-forking conditional-branch exploration over a MachinePathForest.

``MachinePathForest`` (machine_path_forest.py) already implements the fork
tree itself -- possible-world read heads, ``fork_read_head`` for creating
retained sibling worlds with a per-branch ``state_transform``, and
``advance_parallel`` for stepping many heads at once. What it does not do
is decide *when* to fork: ordinary stepping resolves a
``CONDITIONAL_RELATIVE_JUMP`` through ``predicate_handler`` to one taken
boolean, same as any other reversible run.

This module adds the "auto-fork every conditional branch instead of
resolving it" policy on top, without touching ``predicate_handler`` or the
orchestrator at all: it decodes the *next* instruction before stepping
(the same ``_decode_instruction_from_state`` the orchestrator's own
``step`` uses), and when that instruction is a conditional jump, calls
``fork_read_head`` with two branches whose ``state_transform`` sets ``pc``
directly to the fall-through or jump target -- the exact same two
``replace(advanced, pc=...)`` outcomes ``_step_decoded``'s own
``CONDITIONAL_RELATIVE_JUMP`` handling already produces for whichever one
``predicate_handler`` would have picked. Forking both is just "take both."

Head scheduling is DFS/BFS mixed: newly forked heads are appended to a
plain list (no ``collections.deque`` -- everything in this module has to
pass through the same compiler as the machine it drives, and a deque has
no proven lowering there); ``dfs_bias`` controls whether each pick pops from the
end just pushed to (depth-first: keep driving the most recently forked
world deeper) or the far end (breadth-first: cycle through every live
world before returning to any one of them). 1.0 is pure DFS, 0.0 is pure
BFS. Heads are advanced strictly one at a time in this module -- not
through ``advance_parallel``'s ThreadPoolExecutor -- because forking mid-
step requires interrogating and mutating one head's status deterministically
between every instruction; concurrent stepping would race that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Callable

from .machine_execution import MachineExecutionOrchestrator, MachineExecutionStatus
from .machine_path_forest import (
    MachinePathBranch, MachinePathForest, MachinePathHead, MachinePathHeadStatus,
)
from .machine_reference_vocabulary import MachineSemanticToken

# explore_forking_paths' two while loops are the exploration itself, not an
# incidental iteration this compiler is free to dissolve into per-iteration
# dispatch: `remove_loops` defaults to True (the common case for numeric
# kernels with no meaningful loop identity), which forces native_while off
# and makes every loop here a DISPATCH-strategy loop -- unbakeable for a
# whole_program target regardless of what state-effect classification does.
TURING_PAGE = {"remove_loops": False}


@dataclass(frozen=True)
class ForkExplorationPolicy:
    """How aggressively to dive into a just-forked world versus round-robin.

    ``dfs_bias`` is a ratio in [0, 1]: 1.0 always continues the
    most-recently-forked head (pure DFS -- exhaust one path before any
    sibling), 0.0 always continues the least-recently-touched head (pure
    BFS -- advance every live world one step before revisiting any),
    values in between mix the two randomly per pick.
    """

    dfs_bias: float = 0.5
    maximum_forks: int = 64
    maximum_transitions_per_head: int = 10_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.dfs_bias <= 1.0:
            raise ValueError("dfs_bias must be within [0, 1]")
        if self.maximum_forks <= 0 or self.maximum_transitions_per_head <= 0:
            raise ValueError("exploration limits must be positive")


def explore_forking_paths(
    forest: MachinePathForest,
    root_head_id: int,
    *,
    policy: ForkExplorationPolicy = ForkExplorationPolicy(),
    random_source: Callable[[], float] = random.random,
) -> tuple[MachinePathHead, ...]:
    """Auto-fork every conditional branch reached from ``root_head_id``.

    Returns every head that stopped being advanced -- halted, blocked, or
    ran out of its per-head transition budget while still active. Forked
    (non-terminal) heads are not included; look them up via
    ``forest.provenance_graph()`` if the whole tree shape is needed.
    """

    pending: list[int] = [int(root_head_id)]
    terminal: list[int] = []
    forks_made = 0

    while pending:
        if random_source() < policy.dfs_bias:
            head_id = pending.pop()
        else:
            head_id = pending.pop(0)
        head = forest.heads.get(head_id)
        if head is None or head.status is not MachinePathHeadStatus.ACTIVE:
            continue

        transitions = 0
        while transitions < policy.maximum_transitions_per_head:
            state = head.executor.state
            orchestrator = head.executor.executor
            try:
                instruction = orchestrator._decode_instruction_from_state(
                    state, state.pc,
                )
            except Exception:
                instruction = None
            is_branch = (
                instruction is not None
                and instruction.semantic
                is MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP
            )
            if is_branch and forks_made < policy.maximum_forks:
                target = MachineExecutionOrchestrator._relative_target(instruction)
                next_pc = instruction.address + len(instruction.encoded)
                if target is None:
                    # No resolvable jump target -- nothing to fork into;
                    # fall through to an ordinary (blocked) step below.
                    is_branch = False
                else:
                    branches = (
                        MachinePathBranch(
                            "not-taken",
                            state_transform=lambda s, pc=next_pc: replace(s, pc=pc),
                        ),
                        MachinePathBranch(
                            "taken",
                            state_transform=lambda s, pc=target: replace(s, pc=pc),
                        ),
                    )
                    children = forest.fork_read_head(head_id, branches)
                    forks_made += 1
                    pending.extend(child.head_id for child in children)
                    break
            if not is_branch:
                result = head.executor.step_forward()
                transitions += 1
                if result.status is not MachineExecutionStatus.RUNNING:
                    head.status = (
                        MachinePathHeadStatus.HALTED
                        if result.status is MachineExecutionStatus.HALTED
                        else MachinePathHeadStatus.BLOCKED
                    )
                    terminal.append(head_id)
                    break
        else:
            terminal.append(head_id)

    return tuple(forest.heads[identity] for identity in terminal)


__all__ = ["ForkExplorationPolicy", "explore_forking_paths"]
