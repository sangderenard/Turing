"""Finite proof priorities for identity-preserving compiler rewrites.

Physical SSA allocation IDs are payload, never the identity of a decision.
Rules may strengthen a proof, repeat it, or be rejected by a stronger proof.
Equal priorities retain the incumbent and record the rejected challenger.
"""
from dataclasses import dataclass
from typing import Hashable, Mapping


class TransformationConflict(ValueError):
    """Two rules cannot choose a unique normal form for one identity."""


@dataclass(frozen=True)
class TransformationRule:
    name: str
    priority: int


class TransformationLedger:
    def __init__(self, rules: tuple[TransformationRule, ...],
                 successors: Mapping[str, tuple[str, ...]]):
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
        self._decisions: dict[Hashable, tuple[str, Hashable]] = {}
        self._targets: dict[Hashable, object] = {}
        self._successors = {name: tuple(targets) for name, targets in successors.items()}
        self.events: list[dict] = []
        self._rejected: set[tuple] = set()

    def propose(self, identity: Hashable, rule: str, proof: Hashable,
                *, before=None, after=None) -> bool:
        self.rules[rule]  # Reject unregistered rules even on a first decision.
        previous = self._decisions.get(identity)
        if previous is not None:
            old_rule, old_proof = previous
            rank = self.rules[rule].priority
            old_rank = self.rules[old_rule].priority
            if previous == (rule, proof) and after == self._targets[identity]:
                return True
            if rank <= old_rank:
                rejection = (identity, rule, proof, old_rule, old_proof)
                if rejection not in self._rejected:
                    self._rejected.add(rejection)
                    self.events.append(dict(identity=identity, rule=rule,
                        proof=proof, accepted=False, retained=previous,
                        priority=rank, retained_priority=old_rank,
                        reason="incumbent_tie" if rank == old_rank else "stronger_incumbent",
                        before=before, after=after))
                return False
            if rule not in self._successors.get(old_rule, ()):
                raise TransformationConflict(
                    f"Undeclared transformation for {identity!r}: {old_rule} -> {rule}")
        self._decisions[identity] = (rule, proof)
        self._targets[identity] = after
        self.events.append(dict(identity=identity, rule=rule, proof=proof,
                                priority=self.rules[rule].priority,
                                accepted=True, previous=previous,
                                before=before, after=after))
        return True

    def incumbent_target(self, identity: Hashable):
        """Physical payload of the retained decision; excluded from proof identity."""
        return self._targets[identity]

    def incumbent_priority(self, identity: Hashable) -> int | None:
        previous = self._decisions.get(identity)
        return None if previous is None else self.rules[previous[0]].priority


def frame_transformation_ledger() -> TransformationLedger:
    return TransformationLedger((
        TransformationRule("receiver_field", 1),
        TransformationRule("distinct_owner", 2),
        TransformationRule("distinct_result", 3),
    ), {
        "receiver_field": ("distinct_owner", "distinct_result"),
        "distinct_owner": ("distinct_result",),
    })
