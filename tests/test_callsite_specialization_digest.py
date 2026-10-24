"""The callsite-specialization guard reads every graph the loop writes.

Program: ``entry`` constructs ``Box`` and calls ``t.red(n > 1)``.  ``red`` is
a function-table graph, not one of the loop's ``graphs``.  The first round
published ``red``'s ``flag`` descriptor (``changed`` True) while the digest
read only ``graphs``, so the round's state equalled the seed state and the
guard refused: "repeated a completed state at round 1 (period 1)".  The same
refusal stopped the AbstractTensor whole-class compile in ``min``.
"""

from __future__ import annotations

SOURCE = '''
class Box:
    def __init__(self, v):
        self.data = v

    def red(self, flag):
        if flag:
            return 2.0
        return 1.0


def entry(n):
    t = Box(n * 1.5)
    return t.red(n > 1)
'''


def test_callee_descriptor_publication_is_part_of_the_guard_state(tmp_path):
    from src.compiler.fortran_c_shell import compile_ast_fortran_c_shell

    try:
        compile_ast_fortran_c_shell(
            SOURCE, "entry", {}, tmp_path, name="entry", library=True,
            runtime_closure_only=True,
        )
    except Exception as error:  # noqa: BLE001 - only the guard is pinned
        # The road past the planner still refuses by name (a method result
        # on a constructed record); the fixed point itself must settle.
        assert "repeated a completed state" not in str(error)
