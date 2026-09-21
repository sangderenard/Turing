"""What the controller recorded about its own search, kept apart from what
the engines measured.

The old ``Metrics.error_channels`` carried three unrelated kinds of thing
under one name, and the controller's own bookkeeping was the noisiest of
them: ``dt_unresolved``, ``dt_unresolved_attempts``, ``dt_min_retained``,
``dt_min_retained_violation_count``, the four ``superstep_window_*``
entries, ``superstep_iteration_count``, ``superstep_iteration_cap_hit``.

None of those are errors.  No engine publishes them.  Nothing compares them
to a limit.  They are a report the controller writes about the search it
just performed, and putting them in the same dict as engine measurements
caused two concrete defects:

* The controller had to copy-mutate-reassign a dict it did not own
  (``channels = dict(metrics.error_channels or {})`` ... ``metrics.
  error_channels = channels``) at nine separate sites, and one of those
  sites writes a key AFTER the reassignment and only lands because the two
  names alias the same dict.  Owning the record outright removes the whole
  pattern.
* Injected keys changed ``error_channels.length``, so the controller's own
  bookkeeping altered a quantity the compiled ABI measures.

The controller writes fixed value/presence spans on Metrics. The named record
below is their reporting view; neither view changes the engine's measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


# The numerical controller ABI. Names are consumed only by reporting.
CONTROL_NAMES = (
    "dt_unresolved", "dt_unresolved_attempts", "dt_min_retained",
    "dt_min_retained_violation_count", "superstep_window_requested_s",
    "superstep_window_advanced_s", "superstep_window_remaining_s",
    "superstep_iteration_count", "superstep_iteration_cap_hit",
    "dt_unresolved_report",
)


def empty_control():
    from ..tensors import AbstractTensor
    return AbstractTensor.zeros((len(CONTROL_NAMES),))


def control_report(metrics):
    """Named diagnostics at the reporting boundary, preserving absence."""
    return {name: float(value) for name, value, present in zip(
        CONTROL_NAMES, metrics.control_values.tolist(), metrics.control_present.tolist())
        if present}


@dataclass
class ControlDiagnostics:
    """One step's record of what the controller's own search did.

    Every field is optional in the sense that ``None`` means "this did not
    arise", which stays distinct from a recorded zero -- ``dt_unresolved =
    0.0`` means the search resolved, while ``None`` means the question
    never came up.  The controller's own report is read by people and by
    the attempt log, never judged against a limit.
    """

    dt_unresolved: float | None = None
    dt_unresolved_attempts: int | None = None
    dt_min_retained: float | None = None
    dt_min_retained_violation_count: int | None = None
    superstep_window_requested_s: float | None = None
    superstep_window_advanced_s: float | None = None
    superstep_window_remaining_s: float | None = None
    superstep_iteration_count: int | None = None
    superstep_iteration_cap_hit: bool | None = None
    #: Stable rule identities for the bands a step tripped.  Identities,
    #: not prose: the same authored controller must have a fixed native
    #: representation, and a name is resolved for display only.
    soft_reasons: tuple[str, ...] = ()
    rollback_reasons: tuple[str, ...] = ()

    @classmethod
    def from_metrics(cls, metrics):
        values = control_report(metrics)
        values.pop("dt_unresolved_report", None)
        return cls(**values)

    def recorded(self) -> dict[str, Any]:
        """Only what actually arose, for a log line or an attempt record.

        Absent fields are omitted rather than reported as zero, which is
        the distinction ``.get(name, 0.0)`` used to destroy."""
        recorded: dict[str, Any] = {}
        for name, value in vars(self).items():
            if value is None:
                continue
            if isinstance(value, tuple) and not value:
                continue
            recorded[name] = value
        return recorded

    def with_(self, **changes: Any) -> "ControlDiagnostics":
        """A copy carrying these changes.

        The controller updates its report by replacement rather than by
        mutating a dict shared with the engine's metrics -- which is what
        made the old ``channels[...] = ...`` after publication work only by
        aliasing accident."""
        unknown = set(changes).difference(vars(self))
        if unknown:
            raise TypeError(
                f"unknown control diagnostic(s): {sorted(unknown)}"
            )
        return replace(self, **changes)

    def note_soft(self, *reasons: str) -> "ControlDiagnostics":
        return self.with_(
            soft_reasons=(*self.soft_reasons, *(str(r) for r in reasons))
        )

    def note_rollback(self, *reasons: str) -> "ControlDiagnostics":
        return self.with_(
            rollback_reasons=(
                *self.rollback_reasons, *(str(r) for r in reasons)
            )
        )
