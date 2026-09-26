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
from pathlib import Path


#: The repository's program extraction contract; the compiler refuses
#: to lower without one.
CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


def _lower(source: str, name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return lower_ast_source_to_ssa(source, "train", name=name, extraction_contract=CONTRACT)


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

_TUPLE_ASSIGNMENT_ROUND_TRIP = '''
def update(a, b):
    return a + 1.0, b + 2.0

def train(a, b, epochs):
    total = a + b
    for _ in range(epochs):
        a, b = update(a, b)
        total = a + b
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
    prefix = f"carried_{label}".replace("-", "_")
    module, _outputs, _exports = _lower(source, prefix)
    assert module.functions


@pytest.mark.parametrize(
    ("label", "source", "authored_step"),
    [
        ("call-round-trip", _CALL_ROUND_TRIP, lambda v: v - 0.05 * v),
        # The identity helper was the clearest statement of the OLD failure:
        # a body forwarding its own argument scheduled nothing at all.
        ("identity-round-trip", _IDENTITY_ROUND_TRIP, lambda v: v),
    ],
)
def test_a_carried_value_round_tripped_through_a_call_lowers_and_computes(
    label, source, authored_step
):
    """FIXED: the natural spelling of a training loop is available.

    The control program historically had no vocabulary for an authored call
    -- calls were an overlay stitched into the SSA afterwards by lexical
    anchors -- so a loop body whose only content was a call was empty in the
    plan's own language. ``_schedule_loop_callsites`` now makes such a
    callsite a schedulable STATEMENT, the builder lowers it as a placeholder
    at the plan's position, and frame linking fills the callee and bindings
    in place. The emitted body is the aliased-slot form at its purest:
    ``Call [slot] -> slot``, one in-place update through the call.
    """

    from src.compiler.ssa_python_materializer import materialize_ir_module

    prefix = f"carried_{label}".replace("-", "_")
    module, _outputs, _exports = _lower(source, prefix)
    emitted, skipped = materialize_ir_module(module)
    assert skipped == {}
    namespace: dict = {}
    exec(compile(emitted, "<round-trip>", "exec"), namespace)

    compiled = namespace[f"{prefix}__train"]
    # The authored program is the reference, zero-trip included: ``total``
    # starts as ``update(w)`` before the loop, so ``epochs=0`` returns that,
    # not ``w``.  (Stepping ``w`` by hand agreed only with the old exit that
    # returned ``w``'s carried value for ``total``.)
    authored: dict = {}
    exec(compile(source, "<authored>", "exec"), authored)
    for start, epochs in ((2.0, 3), (2.0, 1), (-1.5, 4), (2.0, 0)):
        expected = authored["train"](start, epochs)
        assert compiled(w=start, epochs=epochs) == pytest.approx(
            expected, abs=1e-12
        )


def test_the_count_of_carried_values_is_not_what_limits_it():
    """Two carried values lower; it is the call boundary that does not."""

    module, _outputs, _exports = _lower(_TWO_CARRIED, "carried_pair")
    assert module.functions


def test_tuple_assignment_projects_each_loop_carried_update():
    """Every unpacked target is an authored producer in the loop body."""

    module, _outputs, _exports = _lower(
        _TUPLE_ASSIGNMENT_ROUND_TRIP, "tuple_assignment"
    )
    assert "tuple_assignment__train" in module.functions
    assert not any(
        "loop_carried" in str(shortfall)
        for function in module.functions.values()
        for shortfall in function.metadata.get("lowering_shortfalls", ())
    )


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
    [(1.0, 0.0, 4), (3.0, -1.0, 6), (2.0, 1.0, 3), (2.0, 1.0, 0)],
)
def test_both_carried_values_are_actually_carried(start, second, epochs):
    """FIXED: both loop-carried values now carry across iterations.

    This was a pinned miscompilation: the reducer's lexical environment is
    populated lazily at first read, so a parameter first touched INSIDE the
    loop was absent from the pre-loop snapshot and could never be discovered
    as carried state -- it was passed as its entry value every iteration,
    silently. Parameters read by a loop are now materialized before the
    snapshot (topological_reducer), which is strictly more faithful to
    Python, where a parameter exists from function entry.

    The frozen-reference guard stays: matching the frozen variant again would
    mean the regression returned.
    """

    from src.compiler.ssa_python_materializer import materialize_ir_module

    module, _outputs, _exports = _lower(_TWO_CARRIED_NUMERIC, "twocarry")
    emitted, skipped = materialize_ir_module(module)
    assert skipped == {}
    namespace: dict = {}
    exec(compile(emitted, "<round-trip>", "exec"), namespace)

    produced = namespace["twocarry__train"](w=start, m=second, n=epochs)

    assert produced == pytest.approx(_authored(start, second, epochs), abs=1e-12)
    if epochs:
        assert produced != pytest.approx(
            _with_second_value_frozen(start, second, epochs), abs=1e-9
        )


# -- one value, two bindings ------------------------------------------------
#
# A carried binding that starts from (or updates to) the same value another
# binding holds.  The loop may rebind only the binding it carries; every read
# is resolved through the identity book by the binding it names, never by the
# shared value id.  Each program here miscompiled when reads were keyed by
# value: a loop-body region, a call argument, a call passing one value as two
# operands, the while predicate (a latch that re-read the carried binding for
# ``value`` never terminated), and exits of two bindings sharing an update.

_SHARED_VALUE_PROGRAMS = {
    "region-read": (
        "def train(value, limit):\n"
        "    total = 0.0\n"
        "    second = value\n"
        "    while total < limit:\n"
        "        total = total + value\n"
        "        second = total\n"
        "    return second\n",
        ((1.0, 3.0), (2.0, 7.0), (1.0, 0.0), (5.0, -1.0)),
    ),
    "call-argument": (
        "def helper(a):\n"
        "    return a * 2.0\n\n"
        "def train(value, limit):\n"
        "    total = 0.0\n"
        "    second = value\n"
        "    while total < limit:\n"
        "        total = total + helper(value)\n"
        "        second = total\n"
        "    return second\n",
        ((1.0, 3.0), (2.0, 7.0), (1.0, 0.0)),
    ),
    "call-two-operands": (
        "def helper(a, b):\n"
        "    return a + b\n\n"
        "def train(value, count):\n"
        "    second = value\n"
        "    for _ in range(count):\n"
        "        second = helper(second, value)\n"
        "    return second\n",
        ((1.0, 3), (2.0, 1), (1.5, 0)),
    ),
    "predicate-region": (
        "def helper(a):\n"
        "    return a\n\n"
        "def train(value, limit):\n"
        "    total = 0.0\n"
        "    second = value\n"
        "    while total < value * limit:\n"
        "        total = total + 1.0\n"
        "        second = total\n"
        "    return second\n",
        ((1.0, 3.0), (2.0, 2.0), (1.0, 0.0)),
    ),
    "predicate-direct": (
        "def helper(a):\n"
        "    return a\n\n"
        "def train(value, limit):\n"
        "    total = 0.0\n"
        "    second = value\n"
        "    while total < value:\n"
        "        total = total + limit\n"
        "        second = total\n"
        "    return second\n",
        ((3.0, 1.0), (2.0, 0.5), (0.0, 1.0)),
    ),
    "for-exit": (
        "def helper(a):\n"
        "    return a\n\n"
        "def train(value, count):\n"
        "    second = value\n"
        "    total = 0.0\n"
        "    for _ in range(count):\n"
        "        total = total + value\n"
        "        second = total\n"
        "    return second\n",
        ((1.0, 3), (2.0, 1), (1.5, 0)),
    ),
    "shared-update": (
        "def helper(a):\n"
        "    return a\n\n"
        "def train(w, count):\n"
        "    total = w * 1.0\n"
        "    for _ in range(count):\n"
        "        w = w * 0.5\n"
        "        total = w\n"
        "    return total, w\n",
        ((2.0, 3), (2.0, 0), (-1.0, 2)),
    ),
}


@pytest.mark.parametrize("label", sorted(_SHARED_VALUE_PROGRAMS))
def test_one_value_held_by_two_bindings_computes_the_authored_answer(label):
    import inspect

    from src.compiler.ssa_python_materializer import materialize_ir_module

    source, probes = _SHARED_VALUE_PROGRAMS[label]
    authored_namespace: dict = {}
    exec(compile(source, "<authored>", "exec"), authored_namespace)
    authored = authored_namespace["train"]
    prefix = "shared_" + label.replace("-", "_")
    module, _outputs, _exports = _lower(source, prefix)
    emitted, skipped = materialize_ir_module(module)
    assert skipped == {}
    namespace: dict = {}
    exec(compile(emitted, "<compiled>", "exec"), namespace)
    compiled = namespace[f"{prefix}__train"]
    names = list(inspect.signature(authored).parameters)
    for probe in probes:
        expected = authored(*probe)
        produced = compiled(**dict(zip(names, probe)))
        assert produced == pytest.approx(expected, abs=1e-12), probe
