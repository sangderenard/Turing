"""A loop with real blockers must refuse to compile rather than silently
running its body once with no iteration -- and a raise shaped as an
ordinary guard clause must compile for real instead of being one of
those blockers.

Regression pin for the fix in ``glsl_deployment_strategy.py``'s
``prepare_graph_precompile``: a blocked loop (``control_program is None``)
used to be silently excluded from the composed schedule while its BODY
REGIONS remained in it, unwrapped by any loop construct -- the compiler
reported ``complete: True``, no shortfall, and ran the loop body exactly
once regardless of the requested iteration count (HANDOFF_SHOAL_AND_RE_TARGETS.md
section 6i). ``test_a_bare_raise_inside_a_loop_still_refuses`` pins that a
raise shape nothing recognizes still hits this refusal instead of silently
mis-compiling.

``test_a_guard_clause_raise_inside_a_loop_compiles_and_validates`` pins the
follow-up fix (section 6l): `if cond: body else: raise` is a guard clause,
not a blocker. ``topological_reducer.py``'s ``ast.If`` reduction skips the
Phi merge when exactly one arm is a dead end (raise/return), so the
surviving arm's binding flows through unmerged; ``loop_composer.py``
recognizes the raise-only arm and lowers it to the same ``ValidationBlock``
abort gate already used for the mirror `if cond: raise` shape. If a future
change to either side makes this raise stop being recognized as a
validated guard, this test starts failing with a refusal (or worse, a
silent mis-compile) instead of a clean validated loop.
"""
from __future__ import annotations

import warnings

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


GUARD_CLAUSE_RAISE_SOURCE = (
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

# A raise with no enclosing guard `if` at all -- not the recognized guard
# shape, so it must remain a blocker and hit the orphaned-loop refusal.
BARE_RAISE_SOURCE = (
    "def helper(a):\n    return a * 1.0\n\n"
    "def train(w, n):\n"
    "    total = helper(w)\n"
    "    for k in range(n):\n"
    "        total = total + 1.0\n"
    "        raise ValueError('boom')\n"
    "    return total\n"
)

# Identical shape, benign else instead of raise: proves the guard-clause
# handling is about `raise` specifically, not any if/else inside a loop.
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


def test_a_bare_raise_inside_a_loop_still_refuses():
    with pytest.raises(ValueError) as excinfo:
        _lower(BARE_RAISE_SOURCE, "orphanrefusal")

    message = str(excinfo.value)
    assert "loop's body regions are scheduled but the loop itself" in message
    assert "blockers=('Raise',)" in message


def test_a_guard_clause_raise_inside_a_loop_compiles_and_validates():
    module, outputs, exports = _lower(GUARD_CLAUSE_RAISE_SOURCE, "guardclause")
    fn = module.functions["guardclause__train"]
    # A real composed loop, plus the validation abort gate the raise
    # lowered to -- not silently dropped, not left as an unrecognized
    # blocker.
    assert {
        "loop_header", "loop_body", "loop_latch", "loop_exit",
        "validation_pass", "validation_fail",
    } <= set(fn.blocks)


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
