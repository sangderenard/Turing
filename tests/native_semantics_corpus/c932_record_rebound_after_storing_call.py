# A record passed to a call that stores into it, then the name rebound to a
# fresh construction: the new record was never passed, so its field is the
# constructor's value.  The mark named the binding, so the read refused
# (bootstrap: level = IdentityLogLevel(level); level.name).
from dataclasses import dataclass


@dataclass
class Acc:
    total: int


def add(acc: Acc, v: int) -> int:
    acc.total = acc.total + v
    return acc.total


def root(a: int, b: int) -> int:
    acc = Acc(a)
    add(acc, b)
    acc = Acc(b * 3)
    return acc.total

# INPUTS [{'a': 1, 'b': 2}, {'a': -4, 'b': 5}]
