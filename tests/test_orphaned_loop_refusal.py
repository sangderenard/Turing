"""A loop must retain its authored raise as a native exit.

Regression pin for the fix in ``glsl_deployment_strategy.py``'s
``prepare_graph_precompile``: a blocked loop (``control_program is None``)
used to be silently excluded from the composed schedule while its BODY
REGIONS remained in it, unwrapped by any loop construct -- the compiler
reported ``complete: True``, no shortfall, and ran the loop body exactly
once regardless of the requested iteration count (HANDOFF_SHOAL_AND_RE_TARGETS.md
section 6i). A direct bare raise now has a native validation and a fault
exit, so it no longer needs the blocked-loop refusal.

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

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from pathlib import Path


#: The repository's program extraction contract; the compiler refuses
#: to lower without one.
CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


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

# A direct raise has no guard predicate. Its authored position still has to
# terminate the native loop on the first executed iteration.
BARE_RAISE_SOURCE = (
    "def train(n):\n"
    "    total = 0\n"
    "    for k in range(n):\n"
    "        total = total + 1\n"
    "        raise ValueError('boom')\n"
    "    return total\n"
)

# The empty destructuring target intentionally creates no target binding.
# The iterable remains the exact owner of the runtime loop extent, so the
# compiler must retain a CFG loop around the scheduled body instead of
# calling the bound unresolved.
TARGETLESS_ITERABLE_SOURCE = (
    "def train(total, rows):\n"
    "    for [] in rows:\n"
    "        total = total + 1.0\n"
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
        return lower_ast_source_to_ssa(
            source, "train", name=name, extraction_contract=CONTRACT,
            progress=lambda _message: None, progress_bars=False,
        )


def test_a_bare_raise_inside_a_loop_has_a_native_fault_exit():
    module, _outputs, exports = _lower(BARE_RAISE_SOURCE, "bare_raise")
    fn = module.functions[exports[0]]
    assert {"loop_header", "loop_body", "validation_fail",
            "validation_fault_exit"} <= set(fn.blocks)
    assert any(
        instruction.attributes.get("fault_exit")
        for block in fn.blocks.values()
        for instruction in block.instrs
    )


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


def test_iterable_identity_retains_loop_when_target_has_no_binding():
    module, outputs, exports = _lower(
        TARGETLESS_ITERABLE_SOURCE, "targetless_iterable",
    )
    fn = module.functions["targetless_iterable__train"]

    assert {"loop_header", "loop_body", "loop_latch", "loop_exit"} <= set(
        fn.blocks
    )
    extent, = (
        instruction
        for block in fn.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("binding") == "iterable_extent"
    )
    assert extent.attributes["source_value_id"] in {
        int(value_id) for _name, value_id in fn.metadata["parameter_names"]
    }


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
