"""Literal loop bounds retain every parameter the loop writes to.

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

``B`` is written on every iteration and is not returned.  This formerly let
literal-loop evaporation discard ``B`` and sometimes leak an induction
variable into the ABI.  Multi-carried recurrence preservation now outranks
unrolling: the loop stays semantic SSA, both array publications survive, and
the authored signature remains intact.

Why this matters beyond one kernel: the whole point of compiling authored
Python is that the compiler resolves inefficiencies so nobody has to hand-write
a perfect kernel.  A pass that silently changes a function's signature under
specialisation breaks the premise -- the more you tell the compiler, the less
the result matches what you wrote.  The specialised eigh reached the same
defect twice over: it also hoisted the loop induction variables into the
signature, and its SSA carried an instruction whose operand list the reference
evaluator walked off the end of.

The signature remains the direct observation pinned here.
"""
from __future__ import annotations

import warnings

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


def test_a_literal_bound_keeps_every_authored_parameter():
    function, parameters, values = _lowered(LITERAL_BOUND, "lit")
    assert "B" in parameters, (
        f"B was written on every iteration but is not a parameter: {parameters}"
    )
    assert "B" in values
    assert len(function.args) == 2


def test_literal_bound_records_both_parameters_and_no_counter_formal():

    function, parameters, values = _lowered(LITERAL_BOUND, "shape")
    assert set(parameters) == {"A", "B"}
    assert "B" in values
    assert "i" not in parameters
    assert len(function.args) == 2


WRITE_ONLY_RETURNS_B = """
def f(A, B):
    for i in range(4):
        A[i] = A[i] * 2.0
        B[i] = B[i] + 1.0
    return B
"""

READ_INTO_RESULT = """
def f(A, B):
    for i in range(4):
        A[i] = A[i] * 2.0 + B[i]
    return A
"""

NO_LOOP = """
def f(A, B):
    A[0] = A[0] * 2.0
    B[0] = B[0] + 1.0
    return A
"""


def test_both_mutated_arrays_survive_regardless_of_return_choice():

    _f, returns_a, _v = _lowered(LITERAL_BOUND, "keepa")
    _f, returns_b, _v = _lowered(WRITE_ONLY_RETURNS_B, "keepb")
    assert set(returns_a) == {"A", "B"}
    assert set(returns_b) == {"A", "B"}


def test_a_write_that_feeds_the_result_is_never_dropped():
    """Reading B into the returned value keeps it, which is the tell.

    The eliminated store is not unobservable -- it lands in an array the
    CALLER owns and can read afterwards. It only looks dead from inside the
    function, and that is precisely the reasoning error.
    """

    _function, parameters, _values = _lowered(READ_INTO_RESULT, "readin")
    assert set(parameters) == {"A", "B"}


def test_the_same_writes_without_a_loop_keep_both_parameters():
    """Bounds the defect to the loop path rather than to in-place writes."""

    _function, parameters, _values = _lowered(NO_LOOP, "noloop")
    assert set(parameters) == {"A", "B"}


def test_nested_literal_bound_keeps_parameters_and_bounds_counter_leak():

    function, parameters, values = _lowered(LITERAL_NESTED, "nested")
    assert set(parameters) == {"A", "B"}
    assert "B" in values
    # The write-only parameter loss is fixed.  A dependent inner range still
    # exposes its outer induction capture as one unnamed formal; keep that
    # narrower ABI defect explicit until loop-bound captures are wired to the
    # enclosing Phi rather than promoted at the region boundary.
    formal_ids = {int(argument.id) for argument in function.args}
    assert values.get("j") not in formal_ids
    assert values.get("i") in formal_ids
    assert len(formal_ids - set(parameters.values())) == 1
