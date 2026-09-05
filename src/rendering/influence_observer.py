"""Turn trace records from a running artifact into dye in the field.

This is the join the whole arrangement exists for. The artifact writes the
cheapest true thing it knows -- a site number and a duration -- into a ring it
owns. The manifest, recorded at compile time, says what that site corresponds
to at whichever level a viewer cares about. Here the two meet: a record comes
out of the ring, the manifest resolves it, and dye is released at the nodes
that actually ran.

Nothing about the transport changes. Injected dye enters as arrivals exactly
as emitted dye did, so the solver cannot tell the difference and does not need
to. What changes is only *why* a drop exists: because a region ran, rather
than because a clock came round.

The observer never drives the animation. If the artifact is idle, no dye is
released and the network drains -- which is the correct picture of a program
that is not running, and the one a metronome could never show.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..compiler.influence_field import SPECTRUM_END


class TraceObserver:
    """Drain a trace ring and release dye where the program actually went."""

    def __init__(
        self,
        flow: Any,
        manifest: Mapping[str, Any],
        *,
        level: str = "ssa",
        weight: float = 1.0,
    ) -> None:
        self.flow = flow
        self.manifest = manifest
        self.level = str(level)
        self.weight = float(weight)

        levels = manifest.get("levels") or {}
        if self.level not in levels:
            raise KeyError(
                f"trace manifest has no level {self.level!r}; "
                f"one of {sorted(levels)}"
            )
        self._resolution = levels[self.level]

        # Hue by site, spread along the arc in site order, so colour says
        # *which part of the program* rather than when a clock fired.
        sites = tuple(manifest.get("sites") or ())
        span = max(1, len(sites) - 1)
        self._hue = {
            int(entry["site"]): SPECTRUM_END * index / span
            for index, entry in enumerate(sites)
        }

        # Counters, because an observer that silently resolves nothing looks
        # exactly like a program that is not running.
        self.seen = 0
        self.released = 0
        self.unresolved = 0
        self.unplaced = 0
        self.lost = 0
        self.last_sequence = -1

    def consume(self, records: Sequence[Mapping[str, Any]]) -> int:
        """Release dye for each record. Returns how many drops landed."""

        landed = 0
        for record in records:
            self.seen += 1
            self.lost += int(record.get("lost_before", 0) or 0)
            self.last_sequence = int(record.get("sequence", self.last_sequence))

            site = int(record.get("region", -1))
            targets = self._resolution.get(site)
            if not targets:
                # The artifact reported a site the manifest does not describe.
                # Counted rather than ignored: it means the two were built from
                # different compilations, and the picture would be a lie.
                self.unresolved += 1
                continue

            hue = self._hue.get(site, 0.0)
            # One launch is one release, divided across the values that launch
            # produced, so a region with many results does not out-shout a
            # region with one merely by being larger.
            share = self.weight / len(targets)
            placed = False
            for value_id in targets:
                if self.flow.inject(value_id, hue, share):
                    placed = True
                    landed += 1
            if placed:
                self.released += 1
            else:
                self.unplaced += 1
        return landed

    def summary(self) -> str:
        return (
            f"seen={self.seen} released={self.released} "
            f"unresolved={self.unresolved} unplaced={self.unplaced} "
            f"lost={self.lost} last_seq={self.last_sequence}"
        )


def resolve_field_keys(
    manifest: Mapping[str, Any],
    field: Any,
    *,
    level: str = "ssa",
) -> dict[int, tuple[Any, ...]]:
    """Map each traced site onto the field's own node keys.

    The manifest speaks in value ids; the field keys nodes by component
    reference. This is where the two vocabularies are reconciled, once, so the
    per-record path stays a dictionary lookup.
    """

    levels = manifest.get("levels") or {}
    resolution = levels.get(level) or {}
    by_value: dict[int, list[Any]] = {}
    for key in field._nodes:
        local = getattr(key, "local_id", None)
        if local is None:
            continue
        try:
            by_value.setdefault(int(local), []).append(key)
        except (TypeError, ValueError):
            continue
    return {
        int(site): tuple(
            node
            for value_id in values
            for node in by_value.get(int(value_id), ())
        )
        for site, values in resolution.items()
    }


__all__ = ["TraceObserver", "resolve_field_keys"]
