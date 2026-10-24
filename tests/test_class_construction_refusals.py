"""Ordinary class construction in a program: the refusals that replace
silent wrong answers (doc 40, F3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


def _lower(source: str, name: str):
    return lower_ast_source_to_ssa(
        source, "root", name=name, extraction_contract=str(CONTRACT),
        progress=lambda _message: None,
    )


def test_computed_init_field_without_slot_is_refused():
    # ``self.flag = n > 3`` has no field receipt, so the record view has no
    # slot for it; ``pick`` then read garbage (n=4 gave 3, Python 8).
    with pytest.raises(Exception, match="no physical field for attributes"):
        _lower('''
class C:
    def __init__(self, n: int):
        self.n = n
        self.flag = n > 3
    def pick(self) -> int:
        if self.flag:
            return self.n * 2
        return self.n - 1
def root(n: int) -> int:
    return C(n).pick()
''', "refusal_unslotted_field")


@pytest.mark.parametrize("source", [
    '''
class Acc:
    def __init__(self, start: int):
        self.total = start
    def add(self, v: int) -> int:
        self.total = self.total + v
        return self.total
def root(a: int, b: int) -> int:
    acc = Acc(a)
    acc.add(b)
    return acc.total
''',
    '''
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
    return acc.total
''',
], ids=["method", "dataclass_free_function"])
def test_record_field_read_after_storing_call_is_refused(source):
    # Both answered ``a`` (the construction's value) at main dd340a1: the
    # call's store into the record was dropped with the call.
    with pytest.raises(Exception, match="record_stored_by_call"):
        _lower(source, "refusal_stored_by_call")
