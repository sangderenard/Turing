# A frozen record passed to a call that may store into it (the callee binds
# it to another name), then a field read: no call can rebind a frozen
# record's field, so the read is the construction's value.  It refused
# (bootstrap: SSAObjectContext in validate_context_function).
from dataclasses import dataclass


@dataclass(frozen=True)
class P:
    k: int
    m: int


def g(p: P) -> int:
    q = p
    return q.k + q.m


def root(n: int) -> int:
    p = P(n, n * 2)
    s = g(p)
    return s * 10 + p.k

# INPUTS [{'n': 1}, {'n': -3}]
