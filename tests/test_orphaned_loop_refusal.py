"""A loop with real blockers (e.g. an authored ``raise``) must refuse to
compile rather than silently running its body once with no iteration.

Regression pin for the fix in ``glsl_deployment_strategy.py``'s
``prepare_graph_precompile``: a blocked loop (``control_program is None``)
used to be silently excluded from the composed schedule while its BODY
REGIONS remained in it, unwrapped by any loop construct -- the compiler
reported ``complete: True``, no shortfall, and ran the loop body exactly
once regardless of the requested iteration count, with the raise statement
itself dropped entirely (HANDOFF_SHOAL_AND_RE_TARGETS.md section 6i).

This is also the safety net any future attempt at making raise-carrying
loops actually compose (section 6h/6j) must not defeat by accident -- a
change that makes the loop-composer's blocker stop recognizing a raise
(for example, by giving the raise a value and thereby removing the
``ast.Raise``-typed AST node the blocker scans for) without also making
the loop genuinely compose correctly will silently reopen this exact gap.
If this test starts failing with something OTHER than the expected
ValueError, that is the signal.
"""
from __future__ import annotations

import warnings

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


RAISE_IN_LOOP_SOURCE = (
    "def helper(a):\n    return a * 1.0\n\n"
    "def train(w, n):\n"
    "    total = helper(w)\n"
    "    for k in range(n):\n"
    "        if total > 0.0:\n"
    "            total = total + 1.0\n"
    "        else:\n"
    "            raise ValueError('boom')\n"
    "    return total\n"
)

# Identical shape, benign else instead of raise: proves `raise` specifically
# is what the refusal is about, not merely "a loop with an if/else inside it".
BENIGN_CONTROL_SOURCE = (
    "def helper(a):\n    return a * 1.0\n\n"
    "def train(w, n):\n"
    "    total = helper(w)\n"
    "    for k in range(n):\n"
    "        if total > 0.0:\n"
    "            total = total + 1.0\n"
    "        else:\n"
    "            total = total - 1.0\n"
    "    return total\n"
)


def _lower(source: str, name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return lower_ast_source_to_ssa(source, "train", name=name)


def test_raise_inside_a_loop_refuses_by_name_instead_of_compiling():
    with pytest.raises(ValueError) as excinfo:
        _lower(RAISE_IN_LOOP_SOURCE, "orphanrefusal")

    message = str(excinfo.value)
    assert "loop's body regions are scheduled but the loop itself" in message
    assert "blockers=('Raise',)" in message


def test_the_same_shape_without_raise_compiles_cleanly():
    """Control: an identical loop/if shape with no raise must be
    completely unaffected -- proves the refusal is about `raise`
    specifically, not the mere presence of a conditional inside a loop."""

    module, outputs, exports = _lower(BENIGN_CONTROL_SOURCE, "orphancontrol")
    fn = module.functions["orphancontrol__train"]
    # A real composed loop has these blocks; a silently-flattened one
    # (the defect this refusal replaces) would not.
    assert {"loop_header", "loop_body", "loop_latch", "loop_exit"} <= set(
        fn.blocks
    )
