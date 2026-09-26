from __future__ import annotations

import ast

from src.compiler.assignment_normalization import (
    normalize_destructuring_assignments,
)


def test_flat_unpack_is_one_evaluated_value_and_ordinary_assignments():
    tree = ast.parse("left, right = produce()")

    receipts = normalize_destructuring_assignments(tree)

    assert len(receipts) == 1
    assert ast.unparse(tree).splitlines() == [
        "__turing_assignment_value_0 = produce()",
        "left = __turing_assignment_value_0[0]",
        "right = __turing_assignment_value_0[1]",
    ]


def test_nested_and_starred_unpack_preserve_python_results_and_single_rhs_call():
    source = '''
calls = 0

def produce():
    global calls
    calls += 1
    return ((1, 2), 3, 4, 5)

(left, right), head, *tail = produce()
result = (left, right, head, tail, calls)
'''
    expected: dict = {}
    exec(compile(source, "<authored>", "exec"), expected)

    tree = ast.parse(source)
    normalize_destructuring_assignments(tree)
    produced: dict = {}
    exec(compile(tree, "<normalized>", "exec"), produced)

    assert produced["result"] == expected["result"] == (1, 2, 3, [4, 5], 1)


def test_normalizer_chooses_temporary_names_absent_from_authored_program():
    tree = ast.parse('''
__turing_assignment_value_0 = "authored"
first, second = (1, 2)
''')

    receipts = normalize_destructuring_assignments(tree)

    assert receipts[0].temporary_names == (
        "__turing_assignment_value_1",
        "__turing_assignment_value_2",
    )


def test_flat_literal_rhs_preserves_real_sources_without_synthetic_projections():
    tree = ast.parse(
        "state[:, 0], state[:, 1] = next_position(), next_velocity()"
    )

    receipts = normalize_destructuring_assignments(tree)

    assert receipts[0].temporary_names == (
        "__turing_assignment_value_0",
        "__turing_assignment_value_1",
    )
    assert ast.unparse(tree).splitlines() == [
        "__turing_assignment_value_0 = next_position()",
        "__turing_assignment_value_1 = next_velocity()",
        "state[:, 0] = __turing_assignment_value_0",
        "state[:, 1] = __turing_assignment_value_1",
    ]
