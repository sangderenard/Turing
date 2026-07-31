"""Planner-owned contiguation of scheduled tensor programs.

Contiguation is not tensor ``contiguous()``.  It groups scheduled operations
whose dependencies can be evaluated by one shader invocation without a
device-wide synchronization point.  It also records index transforms that a
later expression lowerer may absorb.  Any dependency that cannot be expressed
that way becomes an explicit phase boundary for the shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Iterable, Sequence

from src.common.tensors.fused_ir import (
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    FusedProgram,
    canonical_elementwise_op,
)


class IndexRelation(str, Enum):
    """How an output invocation reaches an operation's inputs."""

    SAME_INDEX = "same-index"
    REMAPPABLE = "remappable"
    CROSS_INVOCATION = "cross-invocation"


@dataclass(frozen=True)
class ContiguousOperation:
    program_index: int
    operation: str
    relation: IndexRelation
    reason: str


@dataclass(frozen=True)
class ContiguousPhase:
    index: int
    program_indices: tuple[int, ...]
    operations: tuple[ContiguousOperation, ...]
    barrier_before: str | None = None
    barrier_after: str | None = None


@dataclass(frozen=True)
class ContiguousExecutionPlan:
    phases: tuple[ContiguousPhase, ...]

    @property
    def dispatch_count(self) -> int:
        return len(self.phases)

    @property
    def is_single_dispatch(self) -> bool:
        return len(self.phases) <= 1


_REMAPPABLE = frozenset({
    "reshape", "view", "permute", "transpose", "swapaxes",
    "squeeze", "unsqueeze", "flatten",
})
_CROSS_INVOCATION = frozenset({
    "stack", "cat", "concat", "gather", "scatter", "scatter_add",
    "sum", "mean", "prod", "min", "max", "cumsum", "cumprod",
    "matmul", "conv1d", "conv2d",
})
_SOURCE = frozenset({
    "tensor_from_list", "arange", "zeros", "ones", "full",
})


def _extent(meta) -> int | None:
    if meta is None or meta.shape is None:
        return None
    return int(prod(tuple(int(size) for size in meta.shape) or (1,)))


def classify_program(program: FusedProgram, index: int) -> ContiguousOperation:
    """Classify one scheduled program by its invocation-index contract."""

    names = tuple(str(step.op_name) for step in program.steps)
    if not names:
        return ContiguousOperation(
            index, "empty", IndexRelation.SAME_INDEX, "no operations"
        )
    if any(name in _CROSS_INVOCATION for name in names):
        name = next(name for name in names if name in _CROSS_INVOCATION)
        return ContiguousOperation(
            index,
            name,
            IndexRelation.CROSS_INVOCATION,
            f"{name} reads or combines work owned by other invocations",
        )
    if any(name in _REMAPPABLE for name in names):
        name = next(name for name in names if name in _REMAPPABLE)
        return ContiguousOperation(
            index,
            name,
            IndexRelation.REMAPPABLE,
            f"{name} is an index transform that may be absorbed",
        )
    if all(name in _SOURCE for name in names):
        return ContiguousOperation(
            index,
            names[-1],
            IndexRelation.SAME_INDEX,
            "source values are invocation-local or compile-time",
        )
    try:
        all(
            canonical_elementwise_op(name)[0]
            in ELEMENTWISE_UNARY | ELEMENTWISE_BINARY
            for name in names
        )
    except KeyError:
        pass
    else:
        output_extents = {
            _extent((program.meta or {}).get(int(value_id)))
            for value_id in program.outputs.values()
        }
        input_extents = {
            _extent((program.meta or {}).get(int(value_id)))
            for value_id in program.feeds
        }
        non_scalar_inputs = {
            extent for extent in input_extents
            if extent not in (None, 1)
        }
        non_scalar_outputs = {
            extent for extent in output_extents
            if extent not in (None, 1)
        }
        if (
            not non_scalar_inputs
            or not non_scalar_outputs
            or non_scalar_inputs == non_scalar_outputs
        ):
            return ContiguousOperation(
                index,
                "+".join(names),
                IndexRelation.SAME_INDEX,
                "elementwise with only same-index or scalar operands",
            )
    return ContiguousOperation(
        index,
        "+".join(names),
        IndexRelation.CROSS_INVOCATION,
        "index relation is not proven invocation-local",
    )


def contiguate(programs: Sequence[FusedProgram]) -> ContiguousExecutionPlan:
    """Partition scheduled programs into synchronization-safe shader phases.

    Cross-invocation operations are isolated.  A future index-expression
    lowerer may turn a REMAPPABLE or CROSS_INVOCATION operation back into a
    same-phase operation, but absence of that proof never silently removes a
    required dispatch boundary.
    """

    operations = list(
        classify_program(program, index)
        for index, program in enumerate(programs)
    )
    pure_scalar_indices = {
        index
        for index, program in enumerate(programs)
        if program.outputs
        and all(
            _extent((program.meta or {}).get(int(value_id))) == 1
            for value_id in program.outputs.values()
        )
        and all(
            _extent((program.meta or {}).get(int(value_id))) in (None, 1)
            for value_id in program.feeds
        )
        and operations[index].relation is not IndexRelation.CROSS_INVOCATION
    }
    consumers: dict[int, list[int]] = {}
    for consumer_index, program in enumerate(programs):
        for value_id in program.feeds:
            consumers.setdefault(int(value_id), []).append(consumer_index)
    scalar_producers_by_consumer: dict[int, list[int]] = {}
    for producer_index, program in enumerate(programs):
        if all(
            step.op_name == "tensor_from_list"
            for step in program.steps
        ):
            continue
        for value_id in program.outputs.values():
            if _extent((program.meta or {}).get(int(value_id))) != 1:
                continue
            wide_consumers = tuple(
                consumer_index
                for consumer_index in consumers.get(int(value_id), ())
                if any(
                    (_extent((programs[consumer_index].meta or {}).get(
                        int(output_id)
                    )) or 0) > 1
                    for output_id in programs[consumer_index].outputs.values()
                )
            )
            if wide_consumers:
                for consumer_index in wide_consumers:
                    scalar_producers_by_consumer.setdefault(
                        consumer_index, []
                    ).append(producer_index)
                break
    boundary_after = {
        max(producer_indices):
            "scalar result is broadcast across independently "
            "scheduled invocations"
        for producer_indices in scalar_producers_by_consumer.values()
    }
    operations = tuple(
        operation
        for operation in operations
        if operation.program_index not in pure_scalar_indices
    )
    phases: list[ContiguousPhase] = []
    scalar_operations = tuple(
        classify_program(programs[index], index)
        for index in sorted(pure_scalar_indices)
    )
    if scalar_operations:
        phases.append(ContiguousPhase(
            index=0,
            program_indices=tuple(
                operation.program_index for operation in scalar_operations
            ),
            operations=scalar_operations,
            barrier_after="scalar prelude publishes broadcast operands",
        ))
    pending: list[ContiguousOperation] = []

    def flush(*, before=None, after=None):
        if not pending:
            return
        phases.append(ContiguousPhase(
            index=len(phases),
            program_indices=tuple(op.program_index for op in pending),
            operations=tuple(pending),
            barrier_before=before,
            barrier_after=after,
        ))
        pending.clear()

    previous_boundary = (
        "scalar prelude publishes broadcast operands"
        if scalar_operations
        else None
    )
    for operation in operations:
        if operation.relation is IndexRelation.CROSS_INVOCATION:
            flush(before=previous_boundary, after=operation.reason)
            pending.append(operation)
            previous_boundary = operation.reason
            continue
        pending.append(operation)
        boundary = boundary_after.get(operation.program_index)
        if boundary:
            flush(before=previous_boundary, after=boundary)
            previous_boundary = boundary
    flush(before=previous_boundary)
    return ContiguousExecutionPlan(tuple(phases))


__all__ = [
    "ContiguousExecutionPlan",
    "ContiguousOperation",
    "ContiguousPhase",
    "IndexRelation",
    "classify_program",
    "contiguate",
]
