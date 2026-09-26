"""Open a real binary, decode it, and drive it into a forked exploration forest.

The whole sequence, in order, using the real classes each step already
belongs to -- nothing here re-implements or shortcuts any of them:

1. ``BinaryMachineProgram.load_pe`` -- open the subject bytes and decode them
   into a loaded PE image with a real machine core.
2. ``ReversibleMachineExecutor.create`` -- wrap that core's orchestrator and
   initial state in the bidirectional journal (forward/backward stepping).
3. ``MachinePathForest`` -- the forking container the reversible executor's
   read heads live in.
4. ``machine_fork_exploration.explore_forking_paths`` -- the auto-fork
   driving loop: decode the next instruction, fork on every conditional
   branch, advance heads, until every head is halted, blocked, or exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from .binary_machine_program import BinaryMachineProgram
from .machine_execution import ReversibleMachineExecutor
from .machine_fork_exploration import ForkExplorationPolicy, explore_forking_paths
from .machine_path_forest import MachinePathForest, MachinePathHead

# The fork-exploration loops this entrypoint drives (in
# machine_fork_exploration.py) are the exploration itself, not incidental
# iteration -- see that module's own TURING_PAGE declaration for why
# remove_loops must be False here too, on whichever file's source is
# actually handed to the compiler as the entrypoint's own module.
TURING_PAGE = {
   "remove_loops": False,
   "final_fused_reduction": False,
   "file_parameters": {
      "subject": {
         "name": "subject-binary",
         "accept": ".exe,application/vnd.microsoft.portable-executable,application/octet-stream",
         "purpose": "machine-subject",
            "maximum_bytes": 134217728,
      },
   },
}


def open_decode_step_and_fork(
    subject: bytes,
    root_head_id: int = 0,
    *,
    maximum_heads: int = 64,
    policy: ForkExplorationPolicy = ForkExplorationPolicy(),
    random_source: Callable[[], float] = random.random,
) -> tuple[MachinePathHead, ...]:
    """Open ``subject``, decode it, and auto-fork every conditional branch.

    Returns every head that stopped being advanced -- halted, blocked, or
    ran out of its per-head transition budget -- the same contract
    ``explore_forking_paths`` itself returns.
    """

    machine = BinaryMachineProgram.load_pe(subject, maximum_file_size=128 * 1024 * 1024)
    core = machine.machine.cores[0]
    reversible = ReversibleMachineExecutor.create(core.executor, core.state)
    forest = MachinePathForest(reversible, maximum_heads=maximum_heads)
    return explore_forking_paths(
        forest, root_head_id, policy=policy, random_source=random_source,
    )
