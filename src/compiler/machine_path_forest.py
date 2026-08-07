"""Parallel possible-world read heads over immutable reversible machine state.

Heads are not guest threads. A head owns one alternate execution world and a
private reversible journal. Forks retain their parent and explicit path
constraints; physical workers may advance independent heads concurrently.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .machine_execution import (
    MachineExecutionResult,
    MachineExecutionState,
    MachineExecutionStatus,
    ReversibleMachineExecutor,
)


class MachinePathHeadStatus(str, Enum):
    ACTIVE = "active"
    FORKED = "forked"
    BLOCKED = "blocked"
    HALTED = "halted"


@dataclass(frozen=True, slots=True)
class MachinePathConstraint:
    domain: str
    expression: str
    provenance_sequence: int | None = None

    def __post_init__(self) -> None:
        if not self.domain or not self.expression:
            raise ValueError("path constraints require a domain and expression")


MachinePathStateTransform = Callable[[MachineExecutionState], MachineExecutionState]


@dataclass(frozen=True, slots=True)
class MachinePathBranch:
    label: str
    constraints: tuple[MachinePathConstraint, ...] = ()
    state_transform: MachinePathStateTransform | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("machine path branches require a label")


@dataclass(slots=True)
class MachinePathHead:
    head_id: int
    parent_head_id: int | None
    fork_history_position: int
    fork_tape_sequence: int | None
    branch_label: str
    constraints: tuple[MachinePathConstraint, ...]
    executor: ReversibleMachineExecutor
    status: MachinePathHeadStatus = MachinePathHeadStatus.ACTIVE

    @property
    def state(self) -> MachineExecutionState:
        return self.executor.state


@dataclass(frozen=True, slots=True)
class MachinePathAdvance:
    head_id: int
    transitions: int
    result: MachineExecutionResult | None


class MachinePathForest:
    """Persistent fork tree with bounded parallel physical workers."""

    def __init__(
        self,
        root: ReversibleMachineExecutor,
        *,
        tape_sequence: int | None = None,
        maximum_heads: int = 1024,
        exact_state_store=None,
    ) -> None:
        if maximum_heads <= 0:
            raise ValueError("machine path forest needs positive head capacity")
        self.maximum_heads = int(maximum_heads)
        self.exact_state_store = exact_state_store
        root_executor = root.fork()
        self._heads: dict[int, MachinePathHead] = {
            0: MachinePathHead(
                0, None, root.position, tape_sequence, "root", (), root_executor,
            ),
        }
        self._children: dict[int, list[int]] = {0: []}
        self._next_head_id = 1
        if self.exact_state_store is not None:
            self.exact_state_store.create_head(0, root.position, root.state)
            self._configure_exact_history(0, root_executor)

    def _configure_exact_history(
        self, head_id: int, executor: ReversibleMachineExecutor,
    ) -> None:
        if self.exact_state_store is None:
            return
        fallback = executor.cold_history_loader

        def load(start: int, end: int):
            stored = self.exact_state_store.states_range(head_id, start, end)
            if stored or fallback is None:
                return stored
            return fallback(start, end)

        executor.configure_hot_history(
            executor.maximum_hot_states or self.exact_state_store.states_per_segment,
            cold_history_loader=load,
        )

    @property
    def heads(self) -> Mapping[int, MachinePathHead]:
        return MappingProxyType(dict(self._heads))

    def children(self, head_id: int) -> tuple[int, ...]:
        if head_id not in self._heads:
            raise KeyError(f"unknown machine path head {head_id}")
        return tuple(self._children.get(head_id, ()))

    def provenance_graph(self) -> Mapping[str, object]:
        """Serialize the possible-world tree without confusing it with threads."""

        nodes = []
        edges = []
        for identity in sorted(self._heads):
            head = self._heads[identity]
            nodes.append({
                "head_id": head.head_id,
                "kind": "possible-world-read-head",
                "parent_head_id": head.parent_head_id,
                "branch_label": head.branch_label,
                "fork_history_position": head.fork_history_position,
                "fork_tape_sequence": head.fork_tape_sequence,
                "history_position": head.executor.position,
                "status": head.status.value,
                "constraints": [
                    {
                        "domain": item.domain,
                        "expression": item.expression,
                        "provenance_sequence": item.provenance_sequence,
                    }
                    for item in head.constraints
                ],
                "exact_state_segments": (
                    0 if self.exact_state_store is None else len(
                        self.exact_state_store.heads[str(identity)].segments
                    )
                ),
            })
            if head.parent_head_id is not None:
                edges.append({
                    "source": head.parent_head_id,
                    "target": head.head_id,
                    "kind": "fork_read_head",
                    "history_position": head.fork_history_position,
                    "tape_sequence": head.fork_tape_sequence,
                })
        return MappingProxyType({
            "schema": "turing.machine-path-forest.v1",
            "world_axis": "forked-read-head",
            "thread_axis": "independent-not-represented-here",
            "nodes": tuple(nodes),
            "edges": tuple(edges),
        })

    def fork_read_head(
        self,
        head_id: int,
        branches: Sequence[MachinePathBranch],
        *,
        history_position: int | None = None,
        tape_sequence: int | None = None,
    ) -> tuple[MachinePathHead, ...]:
        """Create retained sibling worlds from any existing history state."""

        source = self._heads.get(int(head_id))
        if source is None:
            raise KeyError(f"unknown machine path head {head_id}")
        choices = tuple(branches)
        if len(choices) < 2 or len({item.label for item in choices}) != len(choices):
            raise ValueError("a read-head fork needs at least two uniquely labelled branches")
        if len(self._heads) + len(choices) > self.maximum_heads:
            raise OverflowError("machine path forest head capacity is exhausted")
        position = source.executor.position if history_position is None else int(history_position)
        if not 0 <= position < source.executor.history_length:
            raise IndexError("fork history position is out of range")
        base = source.executor.fork()
        base.seek_history(position)
        base = base.fork()  # discard later states only in the new world
        anchor_state = base.state
        created = []
        for branch in choices:
            executor = base.fork()
            identity = self._next_head_id
            self._next_head_id += 1
            if self.exact_state_store is not None:
                self.exact_state_store.create_head(
                    identity, position, anchor_state,
                    parent_head_id=source.head_id, fork_position=position,
                    constraints=tuple({
                        "domain": item.domain,
                        "expression": item.expression,
                        "provenance_sequence": item.provenance_sequence,
                    } for item in (*source.constraints, *branch.constraints)),
                )
                self._configure_exact_history(identity, executor)
            if branch.state_transform is not None:
                transformed = branch.state_transform(executor.state)
                if not isinstance(transformed, MachineExecutionState):
                    raise TypeError("path branch transform must return MachineExecutionState")
                executor.commit_shell_effect(transformed)
                if self.exact_state_store is not None:
                    self.exact_state_store.append_state(
                        identity, executor.position, executor.state,
                        event="branch_transform",
                        metadata={"branch_label": branch.label},
                    )
            head = MachinePathHead(
                identity,
                source.head_id,
                position,
                source.fork_tape_sequence if tape_sequence is None else tape_sequence,
                branch.label,
                (*source.constraints, *branch.constraints),
                executor,
            )
            self._heads[identity] = head
            self._children.setdefault(source.head_id, []).append(identity)
            self._children[identity] = []
            created.append(head)
        if self.exact_state_store is not None:
            self.exact_state_store.flush()
        source.status = MachinePathHeadStatus.FORKED
        return tuple(created)

    @staticmethod
    def _advance_one(
        head: MachinePathHead,
        maximum_steps: int,
        transition_observer=None,
    ) -> MachinePathAdvance:
        transitions = 0
        result = None
        while transitions < maximum_steps:
            result = head.executor.step_forward()
            transitions += 1
            if transition_observer is not None:
                transition_observer(head, result)
            if result.status is not MachineExecutionStatus.RUNNING:
                head.status = (
                    MachinePathHeadStatus.HALTED
                    if result.status is MachineExecutionStatus.HALTED
                    else MachinePathHeadStatus.BLOCKED
                )
                break
        return MachinePathAdvance(head.head_id, transitions, result)

    def advance_parallel(
        self,
        head_ids: Sequence[int] | None = None,
        *,
        maximum_steps: int,
        maximum_workers: int | None = None,
    ) -> tuple[MachinePathAdvance, ...]:
        """Advance isolated worlds concurrently; no sibling writes are merged."""

        if maximum_steps <= 0:
            raise ValueError("parallel path advance needs a positive step bound")
        identities = tuple(head_ids) if head_ids is not None else tuple(
            identity for identity, head in self._heads.items()
            if head.status is MachinePathHeadStatus.ACTIVE
        )
        heads = []
        for identity in identities:
            head = self._heads.get(int(identity))
            if head is None:
                raise KeyError(f"unknown machine path head {identity}")
            if head.status is not MachinePathHeadStatus.ACTIVE:
                raise ValueError(f"machine path head {identity} is not active")
            heads.append(head)
        if not heads:
            return ()
        with ThreadPoolExecutor(
            max_workers=maximum_workers or len(heads),
            thread_name_prefix="machine-path",
        ) as pool:
            def retain(head, result):
                if self.exact_state_store is None:
                    return
                instruction = result.instruction
                self.exact_state_store.append_state(
                    head.head_id, head.executor.position, result.state,
                    event="forward",
                    metadata={
                        "status": result.status.name,
                        "instruction_address": (
                            None if instruction is None else int(instruction.address)
                        ),
                        "instruction_bytes": (
                            "" if instruction is None else bytes(instruction.encoded).hex()
                        ),
                        "instruction_token": (
                            None if instruction is None else instruction.token.name
                        ),
                        "semantic": (
                            None if instruction is None else instruction.semantic.name
                        ),
                    },
                )

            futures = [
                pool.submit(self._advance_one, head, maximum_steps, retain)
                for head in heads
            ]
            results = tuple(future.result() for future in futures)
        if self.exact_state_store is not None:
            self.exact_state_store.flush()
        return results


__all__ = [
    "MachinePathAdvance", "MachinePathBranch", "MachinePathConstraint",
    "MachinePathForest", "MachinePathHead", "MachinePathHeadStatus",
]
