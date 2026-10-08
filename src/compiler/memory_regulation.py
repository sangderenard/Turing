"""Stage-boundary memory regulation for one compile.

A compile keeps in RAM what its stages need.  This module is the place where
that is *regulated* rather than hoped for:

* ``stage_boundary(label)`` is called where one stage of the lowering ends and
  the next begins.  It reads the process resident set
  (``shell_telemetry.process_rss_bytes``, the one shared helper) and, when the
  work contract declares a ``memory_budget_bytes`` and the resident set is
  above it, releases RECOMPUTABLE items in the declared order of
  ``RELEASE_ORDER`` until it is under the budget.  Each release is a
  ``memory_release_receipt`` row on the compile's book, DERIVED from the
  ``compile_policy`` ``memory_budget_bytes`` cell (the policy is declared on
  the book when the first release reads it, so a compile that never exceeds
  its budget posts nothing and leaves the book the unregulated compile
  leaves).  Nothing is killed or refused.

* ``end_regulation`` runs at the end of the compile.  What it lets go of is
  not a decision but a retention bug fixed at its source -- state keyed to a
  book that no later read can reach (``release_compile_planning_state``) and
  the cyclic garbage the planning stages leave -- so it is unconditional and
  posts no receipt.

A release is admissible here only if dropping the item cannot change the
compile's output: a pure memo that is recomputed on its next miss
(dependency levels, the polymorphic-formal scan) or garbage nothing can
reach (``gc.collect``).  Items whose recomputation would post rows again or
re-plan a callsite (the callsite shell-type cache, the child-signature
results) are NOT releasable mid-compile; they are released when their book's
compile ends.
"""
from __future__ import annotations

import contextvars
import gc
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from .shell_telemetry import process_rss_bytes


def _release_dependency_levels() -> int:
    from .region_scheduling import release_dependency_levels

    return release_dependency_levels()


def _release_polymorphic_formal_scans() -> int:
    from .glsl_deployment_strategy import release_polymorphic_formal_scans

    return release_polymorphic_formal_scans()


def _collect_cyclic_garbage() -> int:
    return gc.collect()


#: The recomputable items a budget breach releases, in this order (cheapest
#: to recompute first; the collection last, so it also frees what the memo
#: releases above it dropped).
RELEASE_ORDER: tuple[tuple[str, Callable[[], int]], ...] = (
    ("dependency_level_memo", _release_dependency_levels),
    ("polymorphic_formal_scan_memo", _release_polymorphic_formal_scans),
    ("cyclic_garbage", _collect_cyclic_garbage),
)


@dataclass
class MemoryRegulator:
    """One compile's regulation: its book, its budget and its readings."""

    book: Any
    budget_bytes: int | None = None
    #: ``(stage label, resident set)`` at every boundary, in order.
    history: list[tuple[str, int]] = field(default_factory=list)
    releases: int = 0
    _report: bool = field(
        default_factory=lambda: bool(os.environ.get("TURING_COMPILE_RSS")),
    )

    def boundary(self, label: str) -> int:
        """Read the resident set at a stage boundary; above the budget,
        release.  Returns the reading (after any release)."""

        resident = process_rss_bytes()
        self.history.append((str(label), resident))
        if self._report:
            print(
                f"[compiler-rss] {label}: {resident / 2**20:.0f} MiB",
                file=sys.stderr, flush=True,
            )
        budget = self.budget_bytes
        if budget is None or resident <= budget:
            return resident
        for item, release in RELEASE_ORDER:
            before = process_rss_bytes()
            released = int(release())
            resident = process_rss_bytes()
            self._post(str(label), item, before, resident, budget, released)
            if resident <= budget:
                break
        return resident

    def _post(
        self, label: str, item: str, before: int, after: int, budget: int,
        released: int,
    ) -> None:
        from .concordance_declarations import (
            COMPILE_POLICY, MEMORY_RELEASE_RECEIPT, MEMORY_REGULATION,
            MemoryRelease, POLICY_DECLARATION,
        )
        from .identity_concordance import Derived, Mode, Novel

        book = self.book
        book.post(
            COMPILE_POLICY, ("memory_budget_bytes",), str(budget),
            stage=MEMORY_REGULATION,
            provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD,
        )
        policy_cell = book.latest_ref(COMPILE_POLICY, ("memory_budget_bytes",))
        book.post(
            MEMORY_RELEASE_RECEIPT, (label, self.releases),
            MemoryRelease(item, before, after, budget, released),
            stage=MEMORY_REGULATION, provenance=Derived((policy_cell,)),
            mode=Mode.CONCORD,
        )
        self.releases += 1


_ACTIVE: contextvars.ContextVar[MemoryRegulator | None] = (
    contextvars.ContextVar("memory_regulator", default=None)
)


def begin_regulation(
    book: Any, *, owned_book: bool = True,
) -> tuple[MemoryRegulator, contextvars.Token, bool]:
    """Open the regulation of a compile on ``book``.

    The budget is the active work contract's ``memory_budget_bytes``.  The
    third result says whether this compile is the OUTERMOST (no regulation
    was already open), which ``end_regulation`` needs: a nested compile
    must not collect or release on behalf of the compile that is still
    running."""

    from .work_contract import active_contract

    regulator = MemoryRegulator(book, active_contract().memory_budget_bytes)
    outermost = _ACTIVE.get() is None
    token = _ACTIVE.set(regulator)
    regulator.boundary("compile: begin")
    return regulator, token, outermost and owned_book


def stage_boundary(label: str) -> None:
    """Mark the end of a lowering stage (a no-op outside a regulated compile)."""

    regulator = _ACTIVE.get()
    if regulator is not None:
        regulator.boundary(label)


def stage_end(label: str) -> None:
    """Mark the end of a stage whose working set is DEAD here (the stage's
    objects are unreachable by the next one): collect the cyclic garbage it
    leaves -- planned shell classes and closures are reference cycles that
    only a full collection frees -- then take the boundary reading."""

    if _ACTIVE.get() is not None:
        gc.collect()
    stage_boundary(label)


def end_regulation(
    regulator: MemoryRegulator, token: contextvars.Token, *,
    releases_book_state: bool,
) -> None:
    """Close the compile's regulation.

    ``releases_book_state`` (the compile owned its book and was the
    outermost): let go of the planning state keyed to that book, which no
    later read can reach, then collect the cyclic garbage the stages leave.
    A compile that resumed its caller's book keeps that state -- the caller's
    next lowering on the same book may hit it."""

    try:
        if releases_book_state:
            from .glsl_deployment_strategy import release_compile_planning_state

            release_compile_planning_state(regulator.book)
            _release_dependency_levels()
            _release_polymorphic_formal_scans()
            gc.collect()
        regulator.boundary("compile: end")
    finally:
        _ACTIVE.reset(token)
