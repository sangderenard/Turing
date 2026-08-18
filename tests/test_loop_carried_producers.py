"""What may produce a loop-carried value, held as fact rather than folklore.

``aot_compile``'s docstring warns that simultaneous tuple assignment breaks
the loop-carried binding analysis. Lowering one-variable-at-a-time cases
through the canonical source compiler locates the wider rule, and it is not
the obvious one. Calls are fine. What fails is the ROUND TRIP: when the
carried value is a call's input and the call's result is bound back to that
same carried name, the region's input and output fuse and the body publishes
no distinct produced value.

The discriminator is that a call on some other value lowers, and a pure
identity helper on the carried value fails -- there is nothing there to
produce. One ordinary operation on the result restores it, because that
forces a real instruction whose result is the carried value.

This matters for authoring. ``w = adam_update(w, g, m, v)`` is the natural
spelling of a training step and is exactly the round trip that does not
lower, while the failure names an internal value id rather than the line
that caused it.
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

_CALL_ROUND_TRIP = '''
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

_CALL_ON_ANOTHER_VALUE = '''
def update(a):
    return a - 0.05 * a

def train(w, seed, epochs):
    total = update(w)
    for _ in range(epochs):
        next_w = update(seed)
        w = next_w
        total = w
    return total
'''

_CALL_THEN_ONE_OPERATION = '''
def update(a):
    return a - 0.05 * a

def train(w, epochs):
    total = update(w)
    for _ in range(epochs):
        stepped = update(w)
        next_w = stepped * 1.0
        w = next_w
        total = w
    return total
'''

_IDENTITY_ROUND_TRIP = '''
def passthrough(a):
    return a

def train(w, epochs):
    total = passthrough(w)
    for _ in range(epochs):
        next_w = passthrough(w)
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
        # Calls are not the problem: this one consumes a value that is not
        # the carried one, and lowers.
        ("call-on-another-value", _CALL_ON_ANOTHER_VALUE),
        # One ordinary operation on the result forces a real instruction whose
        # result IS the carried value, which is enough.
        ("call-then-one-operation", _CALL_THEN_ONE_OPERATION),
    ],
)
def test_a_carried_value_computed_in_the_body_lowers(label, source):
    module, _outputs, _exports = _lower(source, f"carried_{label}")
    assert module.functions


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("call-round-trip", _CALL_ROUND_TRIP),
        # The clearest statement of the cause: a helper that returns its own
        # argument produces nothing, so input and output fuse completely.
        ("identity-round-trip", _IDENTITY_ROUND_TRIP),
    ],
)
def test_a_carried_value_round_tripped_through_a_call_does_not_lower(label, source):
    """The restriction, pinned so it is a known limit and not a surprise.

    If this ever starts passing, the analysis learned to see through the
    input/output fusion -- delete the test and the docstring warning
    together, and the natural spelling of a training loop is available.
    """

    with pytest.raises(Exception) as raised:
        _lower(source, f"carried_{label}")
    assert "loop_carried" in str(raised.value)


def test_the_count_of_carried_values_is_not_what_limits_it():
    """Two carried values lower; it is the call boundary that does not."""

    module, _outputs, _exports = _lower(_TWO_CARRIED, "carried_pair")
    assert module.functions


# -- a second carried value is silently dropped ---------------------------
#
# Found by round-tripping the lowered program back to Python and running it,
# not by reading the IR. It lowers with no shortfall, executes, and returns
# the wrong number -- the failure mode this pipeline keeps producing.

_TWO_CARRIED_NUMERIC = """
def helper(a):
    return a * 1.0

def train(w, m, n):
    total = helper(w)
    for _ in range(n):
        next_m = m * 0.5 + w
        next_w = w - 0.1 * next_m
        m = next_m
        w = next_w
        total = w
    return total
"""


def _authored(w, m, n):
    for _ in range(n):
        next_m = m * 0.5 + w
        next_w = w - 0.1 * next_m
        m, w = next_m, next_w
    return w


def _with_second_value_frozen(w, m, n):
    """What the compiled program actually computes: ``m`` never updates."""

    for _ in range(n):
        w = w - 0.1 * (m * 0.5 + w)
    return w


@pytest.mark.parametrize(
    ("start", "second", "epochs"),
    [(1.0, 0.0, 4), (3.0, -1.0, 6), (2.0, 1.0, 3)],
)
def test_only_the_first_carried_value_is_actually_carried(start, second, epochs):
    """A KNOWN MISCOMPILATION, pinned so it cannot be mistaken for working.

    The source carries two names across the loop. The lowering creates one
    carried phi, and passes the second value's ENTRY value on every iteration.
    Nothing raises; the numbers are simply wrong.

    When this is fixed the assertions below will fail loudly -- which is the
    point. Flip them to ``_authored`` and delete this comment; do not relax
    them.

    ``loop.carried_aliases`` reaches the SSA builder with a single entry, so
    the fix belongs upstream where ``loop_composer`` builds that tuple, not in
    ``precompile_to_ssa``, which lowers faithfully what it is handed.
    """

    from src.compiler.ssa_python_materializer import materialize_ir_module

    module, _outputs, _exports = _lower(_TWO_CARRIED_NUMERIC, "twocarry")
    emitted, skipped = materialize_ir_module(module)
    assert skipped == {}
    namespace: dict = {}
    exec(compile(emitted, "<round-trip>", "exec"), namespace)

    produced = namespace["twocarry__train"](w=start, m=second, n=epochs)

    assert produced == pytest.approx(
        _with_second_value_frozen(start, second, epochs), abs=1e-12
    )
    if epochs:
        assert produced != pytest.approx(_authored(start, second, epochs), abs=1e-9)
