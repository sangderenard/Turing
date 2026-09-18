"""Reusing a loop variable name across sequential loops must not break codegen.

Found while compiling an analytic ``eigh`` backward -- five sequential nested
loops, each written with the obvious ``for i ... for j ... for k`` -- which
emitted with **zero shortfalls** and then failed LLVM's own verifier:

    error: invalid LLVM IR input: Instruction does not dominate all uses!
      %phi.136 = phi ptr [ %value.135, %loop_body.1 ], [ %value.32, %loop_latch.2 ]

The trigger is not nesting depth, loop count, or the accumulator pattern --
each of those compiles cleanly on its own, and four sequential triple-nested
matmul blocks compile fine when their loop variables are named apart. It is
**name reuse**: writing ``for i in range(n)`` in two sequential loops of the
same function makes the emitted phi nodes reference definitions from blocks
that do not dominate them.

That is ordinary Python. Nobody renames a loop counter per loop, and the
matching Python and SSA are both perfectly correct -- the reference evaluator
runs the same program without complaint. Only the emitted IR is malformed.

Two things make this worth pinning rather than working around:

* ``emit_ssa_function_to_llvm`` reports ``shortfalls == ()``. The claim of a
  clean emission is wrong, and only the external verifier catches it. Anything
  that trusts the shortfall count -- as an automated function-to-deployment
  tool would -- concludes success.
* The workaround (rename every loop variable apart) is invisible to a caller
  and impossible to discover from the error, which names LLVM temporaries
  rather than source variables.
"""
from __future__ import annotations

import pathlib
import tempfile
import warnings

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
)


def _block(suffix: str) -> str:
    """One triple-nested accumulate, with its loop variables suffixed."""

    return (
        f"    for i{suffix} in range(n):\n"
        f"        for j{suffix} in range(n):\n"
        f"            acc{suffix} = 0.0\n"
        f"            for k{suffix} in range(n):\n"
        f"                acc{suffix} = acc{suffix} + "
        f"V[i{suffix} * n + k{suffix}] * M[k{suffix} * n + j{suffix}]\n"
        f"            T[i{suffix} * n + j{suffix}] = acc{suffix}\n"
    )


def _emit_and_compile(body: str, name: str):
    source = "def f(V, M, T, n):\n" + body + "    return T\n"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "f", name=name
        )
    artifact = emit_ssa_function_to_llvm(module, f"{name}__f")
    assert artifact.shortfalls == (), (
        "the emitter reports a clean emission in every case here; the defect "
        "is that the IR is invalid anyway"
    )
    compile_artifact(
        artifact, directory=pathlib.Path(tempfile.mkdtemp()) / name
    )


def test_distinct_loop_variable_names_compile():
    """The control, and incidentally the workaround."""

    _emit_and_compile(_block("0") + _block("1"), "distinct2")


def test_distinct_names_compile_at_three_blocks_too():
    _emit_and_compile(_block("0") + _block("1") + _block("2"), "distinct3")


def test_a_single_block_compiles_whatever_its_names():
    _emit_and_compile(_block(""), "single")


@pytest.mark.xfail(
    reason=(
        "known defect: two sequential loops reusing the same loop variable "
        "names emit phi nodes that do not dominate their uses, so the module "
        "fails LLVM verification -- while shortfalls is still ()"
    ),
    strict=True,
)
def test_reused_loop_variable_names_compile():
    _emit_and_compile(_block("") + _block(""), "reused2")


def test_the_failure_is_a_dominance_error_and_not_something_else():
    """Pin the actual diagnostic, so a fix is recognisable as a fix.

    Asserting only "it raises" would also pass if this started failing for an
    unrelated reason -- a missing toolchain, say -- which is exactly the kind
    of false green this file exists to prevent.
    """

    with pytest.raises(RuntimeError) as caught:
        _emit_and_compile(_block("") + _block(""), "reusedmsg")
    assert "does not dominate all uses" in str(caught.value)
    assert "phi" in str(caught.value)
