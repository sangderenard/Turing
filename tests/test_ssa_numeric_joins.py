"""A join of int and float arms is refused where Python sees each arm's class.

The join is carried as one physical class (``ssa_numeric_joins``).  That
is exact only where the joined value meets an operand of that class in
arithmetic (corpus c770-c774).  Each program here reads it where an int
and a float differ, and must be refused by name, never answered.
"""

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


REFUSED = "int-or-float union lowering is not built"

PROGRAMS = {
    "returned_from_the_program": """
def root(n: int, c: int):
    if c == 2:
        s = n
    else:
        s = n * 0.5
    return s
""",
    "callee_result_returned": """
def pick(n, c):
    if c == 2:
        return n
    return n * 0.5

def root(n: int, c: int):
    return pick(n, c)
""",
    "compared": """
def root(n: int, c: int):
    if c == 2:
        s = n
    else:
        s = n * 0.5
    return s > 3
""",
    "optional_payload_mod_int": """
def pick(n, c):
    if c == 1:
        return None
    if c == 2:
        return n
    return n * 0.5

def root(n: int, c: int) -> float:
    s = pick(n, c)
    t = 0.0
    if s is not None:
        t = s % 2
    return t
""",
}


def _lower(source):
    return lower_ast_source_to_ssa(
        source, "root", name="numeric_join",
        extraction_contract="extraction_contracts/program_extraction.yaml",
        progress=lambda _: None,
    )


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_type_visible_numeric_join_is_refused(name):
    with pytest.raises(ValueError, match=REFUSED):
        _lower(PROGRAMS[name])


def test_join_read_by_float_arithmetic_lowers():
    module, _outputs, _exports = _lower("""
def root(n: int, c: int) -> float:
    if c == 2:
        s = n
    else:
        s = n * 0.5
    return s + 0.25
""")
    assert module.metadata["numeric_join_promotions"] >= 1
