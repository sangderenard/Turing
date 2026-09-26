"""The shared state union, each participant's concern mask, and the
vectorized delta that says whose concerns actually moved.

Several participants stepping one coupled system need to know, cheaply and
every step, which of them were invalidated by what the others did.  The
expensive ways to answer that are to track a coupling graph and walk it, or
to make every participant re-query whenever anything moves; both scale with
how entangled the system is, which is exactly the lockstep this is meant to
escape.

The cheap way is structural.  Every participant already declares the state
it touches, so their declarations union into one fixed, ordered index.  A
participant's concerns are then a boolean mask over that index -- "it's my
concern" -- and one delta over the union answers every participant at once:
whatever moved, intersected with each mask, is who has to care.  No graph,
no traversal, no per-participant query.

``examples/llvm_dt_system.py`` already has both halves of this in
structural form: ``column_names_of(pieces)`` builds "every column some
piece reads, in first-appearance order" (the union) and each piece's
``argument_names`` is that piece's concerns (the mask).  This module makes
them first-class and adds the delta.

The comparison is vectorized where the data is and looped where the
metadata is: one tensor compare per union entry, over however many
elements that entry holds, and a plain loop across the handful of entry
names.  The loop is over the schema, never over the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def _as_tensor(value: Any):
    """A comparable tensor view of one state entry.

    Entries arrive as AbstractTensors, numpy arrays, or bare scalars
    depending on which lane produced them; comparison has to mean the same
    thing for all three.
    """
    from ...common.tensors import AbstractTensor

    if isinstance(value, AbstractTensor):
        return value
    return AbstractTensor.tensor(value)


@dataclass(frozen=True)
class ConcernUnion:
    """Every state name any participant touches, in one fixed order.

    The order is first-appearance, matching ``column_names_of`` so that a
    union built here and a column layout built there index the same way.
    """

    names: tuple[str, ...]

    @classmethod
    def of(cls, concerns: Iterable[Iterable[str]]) -> "ConcernUnion":
        ordered: list[str] = []
        seen: set[str] = set()
        for participant_concerns in concerns:
            for name in participant_concerns:
                key = str(name)
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        return cls(tuple(ordered))

    def position(self, name: str) -> int:
        return self.names.index(str(name))

    def mask(self, names: Iterable[str]):
        """This participant's "it's my concern" mask over the union.

        A name the union does not carry is a participant declaring a
        concern nobody -- including itself -- published, which is a
        declaration error rather than an empty mask entry to shrug at.
        """
        wanted = {str(name) for name in names}
        unknown = wanted.difference(self.names)
        if unknown:
            raise KeyError(
                "concerns name state outside the union: "
                f"{sorted(unknown)}"
            )
        from ...common.tensors import AbstractTensor

        return AbstractTensor.tensor([name in wanted for name in self.names])

    def __len__(self) -> int:
        return len(self.names)


def entry_changed(before: Any, after: Any) -> bool:
    """Whether one union entry moved at all.

    Exact inequality, deliberately: this feeds energy accounting and the
    audit of a dilating participant's orthogonality claim, where "it
    changed by a little" is still a change and a tolerance would be a place
    for an unaccounted exchange to hide.
    """
    if before is None or after is None:
        return before is not after
    return bool((_as_tensor(before) != _as_tensor(after)).any().item())


def changed_mask(
    union: ConcernUnion,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
):
    """A boolean vector over the union: which entries moved this step.

    One tensor comparison per entry, over all of that entry's elements at
    once; the loop is across entry NAMES (a handful) and never across the
    data (which may be millions of elements).
    """
    from ...common.tensors import AbstractTensor

    return AbstractTensor.tensor([
        entry_changed(before.get(name), after.get(name))
        for name in union.names
    ])


def touches(changed, mask) -> bool:
    """Whether anything that moved is something this mask claims."""
    return bool((changed * mask).any().item())


def invalidated(
    changed, masks: Mapping[str, Any],
) -> tuple[str, ...]:
    """Every participant whose own concerns moved, in declaration order.

    This is the whole answer the coordinator needs: a participant absent
    from this tuple saw nothing it cares about change, so a closed-form
    (dilated) evaluation of it remains valid and it does not need to be
    stepped or re-queried.
    """
    return tuple(
        name for name, mask in masks.items() if touches(changed, mask)
    )
