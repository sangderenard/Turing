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


@pytest.mark.parametrize("source,has_slot", [
    ('''
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
''', True),
    ('''
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
''', False),
], ids=["method", "dataclass_free_function"])
def test_record_field_read_after_storing_call_uses_declared_slot(source, has_slot):
    # Both answered ``a`` (the construction's value) at main dd340a1: the
    # call's store into the record was dropped with the call.
    from src.compiler.identity_concordance import CorrelationTable
    if not has_slot:
        # The dataclass constructor currently has no class-navigation slot.
        # Its conceptual handle is not a sanctioned mutable field arena.
        with pytest.raises(ValueError, match='record_stored_by_call'):
            _lower(source, 'undeclared_dataclass_reload')
        return
    module, _, exports = _lower(source, "reloaded_stored_by_call")
    assert CorrelationTable.build(module).findings(module) == []
    assert all(instruction.args for block in module.functions[exports[0]].blocks.values()
               for instruction in block.instrs if instruction.op == 'Ret')


_STORED_ALIAS_PRELUDE = '''
from dataclasses import dataclass
@dataclass
class Acc:
    total: int
def add(acc: Acc, v: int) -> int:
    acc.total = acc.total + v
    return acc.total
'''


@pytest.mark.parametrize("body", [
    '''
def root(a: int, b: int) -> int:
    acc = Acc(a)
    old = acc
    add(acc, b)
    return old.total
''',
    '''
def root(a: int, b: int) -> int:
    acc = Acc(a)
    old = acc
    add(acc, b)
    acc = Acc(b * 3)
    return old.total * 100 + acc.total
''',
    '''
def root(a: int, b: int) -> int:
    acc = Acc(a)
    old = acc
    t = 0
    for _ in range(3):
        t = t * 10 + old.total
        add(acc, b)
    return t
''',
], ids=["alias", "alias_then_rebound", "alias_in_loop"])
def test_alias_of_record_stored_by_call_is_refused(body):
    # Each answered the construction's ``a`` for ``old.total`` (silent): the
    # storing-call mark named ``acc``, not the record ``old`` also names.
    with pytest.raises(ValueError, match="record_stored_by_call"):
        _lower(_STORED_ALIAS_PRELUDE + body, "stored_alias_refused")


def test_call_result_is_not_one_returned_construction_among_none_exits():
    # ``make`` returns ``Box(v)`` on one path and ``None`` on another.  The
    # one construction was posted as the class of every call result, and
    # topology reduction refused on the contradicting ``None`` exit
    # (bootstrap ``parse_identity_log_level``).
    source = '''
class Box:
    def __init__(self, k: int) -> None:
        self.k = k
def make(v: int):
    if v > 2:
        return Box(v)
    return None
def root(n: int) -> int:
    b = make(n)
    if b is None:
        return -1
    return b.k * 2
'''
    try:
        _lower(source, "optional_construction_result")
    except Exception as error:  # a later, loud refusal is not this stop
        assert "source call result identity concordance" not in str(error)
