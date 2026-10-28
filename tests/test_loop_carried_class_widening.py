"""A loop-carried scalar whose update is of a wider class than its seed.

Python's name is bool (or int) when the loop runs zero times and int (or
float) otherwise, so the result class depends on the trip count.  Where
the trip count is unknown the product must refuse by name, never return
the seed's narrower class.  Where the loop's own record proves it runs
(a non-empty, unmutated display or a literal range), the exit is exactly
the update's class and the program lowers.
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


UNKNOWN_EXTENT_SUM = '''
def root(t: int) -> float:
    xs = [2.5, t * 1.5]
    xs.pop()
    s = 0
    for v in xs:
        s += v
    return s
'''

TUPLE_PRODUCT = '''
def root(t: int) -> float:
    s = 1
    for v in (t * 0.5, 2.0, 3.5):
        s = s * v
    return s
'''

LITERAL_RANGE = '''
def root(n: bool) -> int:
    acc = n
    for i in range(40):
        acc = acc * 2 + i
    return acc
'''


def _lower(source):
    return lower_ast_source_to_ssa(
        source, "root", name="carried_widening",
        extraction_contract=str(CONTRACT), progress=lambda _: None,
    )


def test_mutated_display_has_no_trip_proof():
    with pytest.raises(ValueError, match="result class depends on trip count"):
        _lower(UNKNOWN_EXTENT_SUM)


@pytest.mark.parametrize(
    "source,dtype",
    [(LIST_SUM, "float64"), (TUPLE_PRODUCT, "float64"),
     (LITERAL_RANGE, "int64")],
)
def test_loop_that_provably_runs_carries_the_update_class(source, dtype):
    module, _outputs, _exports = _lower(source)
    headers = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Phi"
        and instruction.attributes.get("binding") == "loop_carried"
    ]
    assert [str(header.res.dtype) for header in headers] == [dtype]
    assert headers[0].attributes.get("numeric_join_promoted")
