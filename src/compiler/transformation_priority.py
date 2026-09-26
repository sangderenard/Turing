"""Finite proof priorities for identity-preserving compiler rewrites.

Physical SSA allocation IDs are payload, never the identity of a decision.
Rules may strengthen a proof, repeat it, or be rejected by a stronger proof.
Equal priorities retain the incumbent and record the rejected challenger.
"""
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Hashable, Mapping



class TransformationConflict(ValueError):
    """Two rules cannot choose a unique normal form for one identity."""


@dataclass(frozen=True)
class TransformationRule:
    name: str
    priority: int


class _BookEventLog(Sequence):
    """A ledger's events, read live from its ``transformation_event`` rows."""

    def __init__(self, page, scope):
        self._page = page
        self._scope = scope

    def _events(self) -> tuple:
        return tuple(self._page.latest(row) for row in self._page.scope_rows(self._scope))

    def __getitem__(self, index):
        return self._events()[index]

    def __len__(self) -> int:
        return len(self._page.scope_rows(self._scope))

    def __eq__(self, other) -> bool:
        return isinstance(other, (list, tuple, _BookEventLog)) and self._events() == tuple(other)

    def __repr__(self) -> str:
        return repr(list(self._events()))

    def __reduce__(self):
        return (list, (list(self._events()),))


class TransformationLedger:
    """Priority-ordered rewrite decisions, stored on the identity book.

    Page ``transformation_decision`` holds each identity's retained
    ``(rule, proof, target)`` at ``(scope, identity)``, revised on every
    accepted proposal; page ``transformation_event`` holds every accepted
    and rejected proposal in order at ``(scope, serial)``.
    """

    def __init__(self, rules: tuple[TransformationRule, ...],
                 successors: Mapping[str, tuple[str, ...]], *,
                 scope: Hashable = None, book=None):
        from .identity_concordance import current_identity_book

        self.rules = {rule.name: rule for rule in rules}
        if len(self.rules) != len(rules):
            raise ValueError("Duplicate transformation rule")
        # Checking every declared edge against a strict rank proves the legal
        # transition graph acyclic, without enumerating factorial permutations.
        for source, targets in successors.items():
            for target in targets:
                if self.rules[target].priority <= self.rules[source].priority:
                    raise TransformationConflict(
                        f"Non-increasing transformation edge: {source} -> {target}")
        self._successors = {name: tuple(targets) for name, targets in successors.items()}
        self.book = current_identity_book() if book is None else book
        self.scope = self.book.mint_scope(scope or "transformation_ledger")
        self._decision_page = self.book.page("transformation_decision")
        self._event_page = self.book.page("transformation_event")
        self.events = _BookEventLog(self._event_page, self.scope)

    def _decision(self, identity: Hashable):
        return self._decision_page.latest((self.scope, identity))

    def _record_event(self, event: dict) -> None:
        self._event_page.concord((self.scope, len(self.events)), event)

    def propose(self, identity: Hashable, rule: str, proof: Hashable,
                *, before=None, after=None) -> bool:
        self.rules[rule]  # Reject unregistered rules even on a first decision.
        decision = self._decision(identity)
        previous = None if decision is None else (decision[0], decision[1])
        if previous is not None:
            old_rule, old_proof = previous
            rank = self.rules[rule].priority
            old_rank = self.rules[old_rule].priority
            if previous == (rule, proof) and after == decision[2]:
                return True
            if rank <= old_rank:
                rejected = any(
                    not event.get("accepted")
                    and (event.get("identity"), event.get("rule"),
                         event.get("proof"), *event.get("retained"))
                    == (identity, rule, proof, old_rule, old_proof)
                    for event in self.events
                )
                if not rejected:
                    self._record_event(dict(identity=identity, rule=rule,
                        proof=proof, accepted=False, retained=previous,
                        priority=rank, retained_priority=old_rank,
                        reason="incumbent_tie" if rank == old_rank else "stronger_incumbent",
                        before=before, after=after))
                return False
            if rule not in self._successors.get(old_rule, ()):
                raise TransformationConflict(
                    f"Undeclared transformation for {identity!r}: {old_rule} -> {rule}")
        self._decision_page.revise((self.scope, identity), (rule, proof, after))
        self._record_event(dict(identity=identity, rule=rule, proof=proof,
                                priority=self.rules[rule].priority,
                                accepted=True, previous=previous,
                                before=before, after=after))
        return True

    def incumbent_target(self, identity: Hashable):
        """Physical payload of the retained decision; excluded from proof identity."""
        decision = self._decision(identity)
        if decision is None:
            raise KeyError(identity)
        return decision[2]

    def incumbent_priority(self, identity: Hashable) -> int | None:
        decision = self._decision(identity)
        return None if decision is None else self.rules[decision[0]].priority


def frame_transformation_ledger(scope: Hashable = None) -> TransformationLedger:
    return TransformationLedger((
        TransformationRule("linked_record_member", 1),
        TransformationRule("distinct_owner", 2),
        TransformationRule("exact_argument_binding", 3),
        TransformationRule("distinct_result", 4),
    ), {
        "linked_record_member": (
            "distinct_owner", "exact_argument_binding", "distinct_result",
        ),
        "distinct_owner": ("exact_argument_binding", "distinct_result"),
        "exact_argument_binding": ("distinct_result",),
    }, scope=scope or "frame_transformation")
