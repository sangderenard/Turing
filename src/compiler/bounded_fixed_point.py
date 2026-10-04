"""One deterministic guard for every ``while changed:`` compiler loop.

A fixed-point loop here must end by construction: it carries a structural
bound derived from the size of the program it runs over, posts one receipt
per round on the concordance (``FIXED_POINT_ROUND``, through
``IdentityBook.post``; round N is DERIVED from round N-1), and refuses with
``ConcordanceRefusal`` when it exhausts the bound or revisits a state.  Until
the bound is hit, behaviour is the caller's own loop, unchanged.

Adoption::

    guard = BoundedFixedPoint(
        "frame-fixed-point", bound_for("frame", functions=len(fns),
                                       call_operands=n_ops, records=n_rec),
        scope=("whole-program",), progress=progress, state=signature(),
    )
    changed = True
    while changed:
        changed, ids = one_round()              # the loop's existing body
        changed = guard.round(changed, changed_ids=ids, state=signature())
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable

from .concordance_declarations import (
    FIXED_POINT_ROOT, FIXED_POINT_ROUND, FIXED_POINT_ROUND_STAGE,
)
from .identity_concordance import (
    ConcordanceRefusal, Derived, Mode, Novel, current_identity_book,
)

#: How many changed ids a receipt row and a message carry.
RECEIPT_IDS = 8

#: Bound formulas, one per audited loop kind.  Every one ends in ``+ 1``: the
#: round that observes "no change" is counted.
_BOUND_FORMULAS = {
    "frame": lambda a: a["functions"] + a["call_operands"] + a["records"] + 1,
    "planner": lambda a: a["nodes"] + a["callsites"] + 1,
    "abi": lambda a: a["functions"] + a["call_operands"] + 1,
}


def bound_for(
    kind: str, *, functions: int = 0, call_operands: int = 0, nodes: int = 0,
    callsites: int = 0, records: int = 0,
) -> int:
    """The structural round bound for a loop of ``kind``.

    ``frame``: functions + call_operands + records + 1.
    ``planner`` (fold / specialization): nodes + callsites + 1.
    ``abi`` (settlement): functions + call_operands + 1.
    Any other kind: the sum of every size given, plus 1 (the default margin).
    """
    sizes = {
        "functions": functions, "call_operands": call_operands,
        "nodes": nodes, "callsites": callsites, "records": records,
    }
    formula = _BOUND_FORMULAS.get(kind)
    total = formula(sizes) if formula else sum(sizes.values()) + 1
    return max(1, int(total))


def _digest(state: Any) -> str:
    if state is None:
        return ""
    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()[:16]


class BoundedFixedPoint:
    """Round counter, receipt writer and refusal for one guarded loop."""

    def __init__(
        self, name: str, bound: int, *, scope: Any = (),
        progress: Callable[[str], None] | None = None, state: Any = None,
        recurrence: str = "refuse", book: Any = None,
    ) -> None:
        if recurrence not in ("refuse", "record"):
            raise ConcordanceRefusal(
                f"{name}: recurrence must be 'refuse' or 'record', "
                f"got {recurrence!r}"
            )
        self.name = str(name)
        self.bound = max(1, int(bound))
        self.scope = scope if isinstance(scope, tuple) else (scope,)
        self.recurrence = recurrence
        self._report = progress or (lambda _message: None)
        self._book = book
        self.rounds = 0
        self._previous_ref = None
        # Round 0 is the state the loop started from (the reference loop,
        # ssa_call_input_adapters, seeds ``seen_states`` the same way), so
        # a first round that restores it has period 1.
        self._seen: dict[str, int] = {}
        digest = _digest(state)
        if digest:
            self._seen[digest] = 0

    def round(
        self, changed: Any, *, changed_ids: Iterable[Any] = (),
        state: Any = None,
    ) -> Any:
        self.rounds += 1
        index = self.rounds
        ids = tuple(changed_ids)
        if isinstance(changed, int) and not isinstance(changed, bool):
            count = changed
        else:
            count = len(ids) or int(bool(changed))
        shown = ids[:RECEIPT_IDS]
        digest = _digest(state)
        period = 0
        if changed and digest:
            earlier = self._seen.get(digest)
            if earlier is not None:
                period = index - earlier
        if digest:
            self._seen[digest] = index

        book = (
            self._book if self._book is not None else current_identity_book()
        )
        row = (self.name, self.scope, index)
        fact = (bool(changed), digest, shown, period)
        provenance = (
            Derived((self._previous_ref,)) if self._previous_ref is not None
            else Novel(FIXED_POINT_ROOT, ())
        )
        self._previous_ref = book.post(
            FIXED_POINT_ROUND, row, fact, stage=FIXED_POINT_ROUND_STAGE,
            provenance=provenance, mode=Mode.REVISE,
        )
        self._report(
            f"{self.name} round {index}/{self.bound} "
            f"changed={count} ids={list(shown)}"
        )
        if period and self.recurrence == "refuse":
            raise ConcordanceRefusal(
                f"{self.name} scope={self.scope!r} repeated a completed state "
                f"at round {index} (period {period}); last changed ids "
                f"{list(shown)}"
            )
        if changed and index >= self.bound:
            raise ConcordanceRefusal(
                f"{self.name} did not converge after {self.bound} rounds; "
                f"last changed ids {list(shown)}"
            )
        return changed
