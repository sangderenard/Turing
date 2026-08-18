"""What may produce a loop-carried value, held as fact rather than folklore.

``aot_compile``'s docstring warns that simultaneous tuple assignment breaks
the loop-carried binding analysis. That is one instance of a wider rule found
by lowering one-variable-at-a-time cases through the canonical source
compiler: the analysis cannot see across a CALL boundary either.

This matters for authoring, not just for the compiler. The natural spelling of
a training loop -- ``w = adam_update(w, g, m, v)`` -- is precisely the form
that does not lower, and the failure names an internal value id rather than
the line that caused it, so it is hard to read backwards.
"""
from __future__ import annotations

import warnings

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


def _lower(source: str, name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return lower_ast_source_to_ssa(source, "train", name=name)


# Each source keeps one other function so the entrypoint is not a bare
# top-level function, which hits an unrelated failure in the graph coordinator.
_INLINE = '''
def helper(a):
    return a * 1.0

def train(w, epochs):
    total = helper(w)
    for _ in range(epochs):
        next_w = w - 0.05 * w
        w = next_w
        total = w
    return total
'''

_TENSOR_METHOD = '''
def helper(a):
    return a * 1.0

def train(w, epochs):
    total = helper(w)
    for _ in range(epochs):
        next_w = (w * 0.95).tanh()
        w = next_w
        total = w
    return total
'''

_TWO_CARRIED = '''
def helper(a):
    return a * 1.0

def train(w, m, epochs):
    total = helper(w)
    for _ in range(epochs):
        next_m = 0.9 * m + 0.1 * w
        next_w = w - 0.05 * next_m
        m = next_m
        w = next_w
        total = w
    return total
'''

_CALL_RESULT = '''
def update(a):
    return a - 0.05 * a

def train(w, epochs):
    total = update(w)
    for _ in range(epochs):
        next_w = update(w)
        w = next_w
        total = w
    return total
'''


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("inline-arithmetic", _INLINE),
        ("tensor-method", _TENSOR_METHOD),
        ("two-carried-values", _TWO_CARRIED),
    ],
)
def test_a_carried_value_computed_in_the_body_lowers(label, source):
    module, _outputs, _exports = _lower(source, f"carried_{label}")
    assert module.functions


def test_a_carried_value_produced_by_a_call_does_not_lower():
    """The restriction, pinned so it is a known limit and not a surprise.

    If this ever starts passing, the analysis learned to see across a call
    boundary -- delete the test and the docstring warning together, and the
    natural spelling of a training loop becomes available.
    """

    with pytest.raises(Exception) as raised:
        _lower(_CALL_RESULT, "carried_call_result")
    assert "loop_carried" in str(raised.value)


def test_the_count_of_carried_values_is_not_what_limits_it():
    """Two carried values lower; it is the call boundary that does not."""

    module, _outputs, _exports = _lower(_TWO_CARRIED, "carried_pair")
    assert module.functions
