"""A literal loop bound must not delete a parameter the loop writes to.

Found while compiling a fully-specialised eigh -- a Jacobi kernel with its
size and sweep count written as literals rather than passed in, so the
compiler had every bound as a constant.  Specialising is supposed to give the
compiler *more* information; here it silently gave the emitted function a
different signature than the author wrote.

    def f(A, B):
        for i in range(4):          # literal bound
            A[i] = A[i] * 2.0
            B[i] = B[i] + 1.0
        return A

``B`` is written on every iteration and is not returned.  With a literal
bound the emitted function's formals are ``[A]`` alone: ``B`` is absent from
``args``, from ``parameter_names`` and from ``value_names``, so no caller can
supply it and its writes have nowhere to land.  Change the single literal to a
parameter (``range(n)``) and all three formals appear correctly, which is what
makes this a defect of specialisation rather than of loops or of in-place
writes.

Why this matters beyond one kernel: the whole point of compiling authored
Python is that the compiler resolves inefficiencies so nobody has to hand-write
a perfect kernel.  A pass that silently changes a function's signature under
specialisation breaks the premise -- the more you tell the compiler, the less
the result matches what you wrote.  The specialised eigh reached the same
defect twice over: it also hoisted the loop induction variables into the
signature, and its SSA carried an instruction whose operand list the reference
evaluator walked off the end of.

Nothing here asserts a wrong numeric answer, because the parameter cannot be
passed at all -- the signature IS the observation.
"""
from __future__ import annotations

import warnings

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


LITERAL_BOUND = """
def f(A, B):
    for i in range(4):
        A[i] = A[i] * 2.0
        B[i] = B[i] + 1.0
    return A
"""

PARAMETER_BOUND = """
def f(A, B, n):
    for i in range(n):
        A[i] = A[i] * 2.0
        B[i] = B[i] + 1.0
    return A
"""

LITERAL_NESTED = """
def f(A, B):
    for i in range(3):
        for j in range(i + 1, 4):
            A[i * 4 + j] = A[i * 4 + j] * 2.0
            B[i * 4 + j] = B[i * 4 + j] + 1.0
    return A
"""


def _lowered(source: str, name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "f", name=name
        )
    function = module.functions[f"{name}__f"]
    return (
        function,
        {str(k): int(v) for k, v in (function.metadata.get("parameter_names") or ())},
        {str(k): int(v) for k, v in (function.metadata.get("value_names") or ())},
    )


def test_a_parameterised_bound_keeps_every_authored_parameter():
    """The control. Identical body, one bound passed in instead of written."""

    function, parameters, _values = _lowered(PARAMETER_BOUND, "ctl")
    assert set(parameters) == {"A", "B", "n"}
    assert len(function.args) == 3


@pytest.mark.xfail(
    reason=(
        "known defect: with a literal loop bound, an array parameter the loop "
        "writes to but does not return is dropped from the emitted signature "
        "entirely -- absent from args, parameter_names and value_names"
    ),
    strict=True,
)
def test_a_literal_bound_keeps_every_authored_parameter():
    function, parameters, values = _lowered(LITERAL_BOUND, "lit")
    assert "B" in parameters, (
        f"B was written on every iteration but is not a parameter: {parameters}"
    )
    assert "B" in values
    assert len(function.args) == 2


def test_the_loss_has_exactly_the_shape_this_file_claims():
    """Pin what actually happens, so a fix is recognisable as a fix.

    Asserting only "B is missing" would also pass if the lowering started
    failing in some entirely different way; asserting that A survives intact
    alongside it keeps this specific.
    """

    function, parameters, values = _lowered(LITERAL_BOUND, "shape")
    assert set(parameters) == {"A"}
    assert "B" not in values
    assert len(function.args) == 1


def test_a_nested_literal_bound_also_leaks_its_induction_variable():
    """The second half of the same defect, at the shape eigh actually uses.

    A dependent inner bound under specialisation both drops ``B`` and hoists
    the inner loop's own counter into the signature, so the emitted function
    takes a formal that is not a parameter and is missing one that is.
    """

    function, parameters, values = _lowered(LITERAL_NESTED, "nested")
    assert set(parameters) == {"A"}
    assert "B" not in values
    # ``j`` is the inner loop's counter, not anything the author passes.
    assert "j" in values
    assert values["j"] in {int(argument.id) for argument in function.args}
