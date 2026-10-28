"""A join of int and float arms is refused where Python sees each arm's class.

The join is carried as one physical class (``ssa_numeric_joins``).  That
is exact only where the joined value meets an operand of that class in
arithmetic (corpus c770-c774).  Each program here reads it where an int
and a float differ, and must be refused by name, never answered.  A join
that leaves as its value-union frame keeps each class (corpus c804-c809),
and so does one read by a region the tag dispatches per arm (c890-c898).
"""

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


REFUSED = "int-or-float union lowering is not built"

PROGRAMS = {
    "carried_read_with_another_carried_read": """
def root(t: int):
    xs = [2.5, t * 1.5]
    s = 0
    r = 0
    for v in xs:
        r = r + s
        s += v
    return r
""",
    "carried_read_by_floordiv": """
def root(t: int) -> float:
    xs = [2.5, t * 1.5]
    s = 0
    q = 0.0
    for v in xs:
        q = q + s // 2
        s += v
    return s + q
""",
    "conditional_update_in_display_loop": """
def root(t: int):
    xs = [2.5, t * 1.5]
    s = 0
    for v in xs:
        if v > 3.0:
            s += v
    return s
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
    # Each arm reads the join with its own class (per-arm region copies;
    # the answers are checked by corpus c770 and c890-c898).
    _lower("""
def root(n: int, c: int) -> float:
    if c == 2:
        s = n
    else:
        s = n * 0.5
    return s + 0.25
""")


CONTINUE_UPDATE = """
def root(t: int) -> float:
    xs = [2.5, t * 1.5]
    s = 0
    for v in xs:
        if v < 3.0:
            continue
        s += v
    return s + 0.0
"""


def test_update_after_continue_is_refused_not_truncated():
    # Python: 4.5 (s is an int on the skipped trip, a float after).
    with pytest.raises(ValueError, match="union lowering is not built"):
        _lower(CONTINUE_UPDATE)
