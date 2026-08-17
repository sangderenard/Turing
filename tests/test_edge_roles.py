"""The operator that recovers argument position from an unordered edge set.

A node's ``parents`` is a set of ``(node, role)``. Position lives in the
role, not in the iteration order, so any consumer that zips or slices
``parents`` is asserting something the graph never promised. These pin the
translation that lets positional code stay positional and stay correct.
"""
from __future__ import annotations

from src.transmogrifier.graph.edge_roles import (
    is_positional,
    keyword_argument_name,
    keyword_arguments,
    ordered_arguments,
    positional_argument_index,
)


def test_both_process_graph_spellings_normalise():
    """``arg:0`` and ``arg0`` are the same statement and must read alike.

    Two spellings existed and were parsed inline in a dozen places, each
    slightly differently. A consumer must never have to know there were
    two.
    """
    assert positional_argument_index("arg:0") == 0
    assert positional_argument_index("arg0") == 0
    assert positional_argument_index("arg:11") == 11
    assert positional_argument_index("arg11") == 11


def test_a_non_positional_role_is_none_not_zero():
    """None means "not an argument"; conflating it with 0 puts a receiver
    or a callee reference into argument zero."""
    for role in ("callee", "func", "value", "lhs", "self", "kw:alpha", ""):
        assert positional_argument_index(role) is None, role
        assert not is_positional(role), role


def test_arguments_come_back_in_source_order_from_any_iteration_order():
    """The whole point: order comes from the role, not from the set."""
    shuffled = [
        (99, "arg:2"),
        (77, "callee"),
        (11, "arg:0"),
        (55, "kw:alpha"),
        (33, "arg1"),
    ]
    assert ordered_arguments(shuffled) == (11, 33, 99)
    # Reversing the input must not change the answer.
    assert ordered_arguments(list(reversed(shuffled))) == (11, 33, 99)


def test_non_positional_edges_do_not_occupy_a_position():
    """A callee reference or keyword must not shift the arguments along."""
    parents = [(7, "callee"), (8, "arg:0"), (9, "kw:beta"), (10, "self")]
    assert ordered_arguments(parents) == (8,)
    assert keyword_arguments(parents) == {"beta": 9}


def test_keyword_names_round_trip():
    assert keyword_argument_name("kw:alpha") == "alpha"
    assert keyword_argument_name("kw:") is None
    assert keyword_argument_name("arg:0") is None


def test_the_compiler_helper_delegates_rather_than_reimplements():
    """One implementation, not two that can drift.

    glsl_deployment_strategy kept its own copy of this parser. A position
    is not a thing to be slightly wrong about, so the name remains for its
    callers but the behaviour must be the graph package's.
    """
    from src.compiler.glsl_deployment_strategy import (
        _positional_argument_index,
    )

    for role in ("arg:0", "arg3", "arg:12", "callee", "lhs", "kw:x", ""):
        assert _positional_argument_index(role) == positional_argument_index(
            role
        ), role
