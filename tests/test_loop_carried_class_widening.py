"""A loop-carried scalar whose update is of a wider class than its seed.

Python's name is bool (or int) when the loop runs zero times and int (or
float) otherwise, so the result class depends on the trip count.  The
product must refuse by name, never return the seed's narrower class.
"""

from pathlib import Path

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "extraction_contracts" / "program_extraction.yaml"

BOOL_INT = '''
def root(n: bool, k: int) -> int:
    acc = n
    for i in range(k):
        acc = acc * 2 + i
    return acc
'''

INT_FLOAT = '''
def root(n: int, k: int) -> float:
    acc = n
    for i in range(k):
        acc = acc * 0.5 + i
    return acc
'''

LIST_SUM = '''
def root(t: int) -> float:
    xs = [2.5, t * 1.5]
    s = 0
    for v in xs:
        s += v
    return s
'''

FIELD_SUM = '''
class Acc:
    def __init__(self):
        self._c = [2.5]
    def total(self) -> float:
        s = 0
        for v in self._c:
            s += v
        return s

def root(t: int) -> float:
    return Acc().total() + t
'''


@pytest.mark.parametrize(
    "source,joined",
    [
        (BOOL_INT, "bool and int"),
        (INT_FLOAT, "int and float"),
        (LIST_SUM, "int and float"),
        (FIELD_SUM, "int and float"),
    ],
)
def test_carried_class_widening_is_refused_by_name(source, joined):
    with pytest.raises(ValueError) as refusal:
        lower_ast_source_to_ssa(
            source, "root", name="carried_widening",
            extraction_contract=str(CONTRACT), progress=lambda _: None,
        )
    text = str(refusal.value)
    assert f"loop-carried value joins {joined} arms" in text
    assert "result class depends on trip count" in text
