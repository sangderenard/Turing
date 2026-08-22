"""A mixed-domain scalar operation promotes; it never demotes.

The lowering spells negation as ``Mul(int -1, x)``. Emission used to take a
binary operation's arithmetic domain from ``args[0]`` alone, so that ran as
an INTEGER multiply and the float operand was reached by ``fptosi``. Every
fractional value truncated to zero -- which raised nothing, dropped no
instruction, and verified clean. It simply deleted terms from expressions.

In the fluid step it removed six of twenty-eight inputs from ``height_next``
and kept the tracer from ever going negative, so the compiled step accepted
a frame the authored mathematics rejects.

These hold the rule directly: if any operand is floating point, the
operation is floating point. The feed is chosen so that truncation is
unmistakable rather than merely inexact -- every value has magnitude below
one, so ``fptosi`` sends it to zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


#: Fractions only. An integer domain sends every one of them to zero, so a
#: demoting emission cannot coincidentally agree.
FEED = (0.000892, 0.004255, -0.001418, 0.5)


def _binary_with_integer_constant(
    operation: str, constant: int, *, constant_first: bool,
) -> tuple:
    """``operation`` over an int constant and a float64 scalar.

    Built as a CALLER plus a callee, because that is the emission path the
    defect lived on: a planner region reached through the repository call
    closure. A single-block root goes down a different path entirely, and a
    test shaped that way passes against the broken backend -- which this
    one was, until it was checked against the pre-fix revision.
    """
    inner_source = SSAValue(1000, dtype="float64", shape=())
    literal = SSAValue(1001, dtype="int", shape=())
    inner_result = SSAValue(1002, dtype="float64", shape=())
    operands = [literal, inner_source] if constant_first else [
        inner_source, literal
    ]
    callee = Function(
        "region",
        [inner_source],
        {
            "entry": BasicBlock("entry", [
                Instr("Const", [], literal, attributes={"value": constant}),
                Instr(operation, operands, inner_result),
                Instr("Ret", [inner_result], None),
            ])
        },
    )

    source = SSAValue(2000, dtype="float64", shape=())
    aggregate = SSAValue(2001, dtype="float64", shape=())
    index = SSAValue(2002, dtype="int", shape=())
    address = SSAValue(2003, dtype="float64", shape=())
    result = SSAValue(2004, dtype="float64", shape=())
    caller = Function(
        "program",
        [source],
        {
            "entry": BasicBlock("entry", [
                Instr(
                    "Call", [source], aggregate,
                    attributes={
                        "callee": "region",
                        "output_ids": (int(inner_result.id),),
                    },
                ),
                Instr("Const", [], index, attributes={"value": 0}),
                Instr(
                    "GetElementPtr", [aggregate, index], address,
                    attributes={"aggregate_index": 0},
                ),
                Instr("Load", [address], result),
                Instr("Ret", [result], None),
            ])
        },
    )
    return IRModule({"program": caller, "region": callee}), source, result


def _run(module, source, result, value: float, directory) -> float:
    artifact = emit_ssa_function_to_llvm(module, "program")
    assert artifact.shortfalls == (), artifact.shortfalls
    native = compile_artifact(artifact, directory=directory)
    execution = prepare_artifact_execution(
        native, {int(source.id): np.asarray([value], dtype=np.float64)},
    )
    execution.run()
    return float(
        np.asarray(execution.buffers[int(result.id)], dtype=float).reshape(-1)[0]
    )


@pytest.mark.parametrize("value", FEED)
@pytest.mark.parametrize("constant_first", [True, False])
def test_negation_by_integer_constant_keeps_the_fraction(
    value, constant_first, tmp_path,
):
    """``Mul(-1, x)`` is negation, not truncation -- from either side.

    Both operand orders matter: taking the domain from args[0] made the
    result depend on which side the integer landed on, which is not a
    property arithmetic has.
    """
    module, source, result = _binary_with_integer_constant(
        "Mul", -1, constant_first=constant_first,
    )
    produced = _run(module, source, result, value, tmp_path / "negate")
    assert produced == pytest.approx(-value, rel=0, abs=0), (
        f"Mul(-1, {value}) produced {produced!r}; a zero here means the "
        "operation ran in the integer domain and truncated the operand"
    )


@pytest.mark.parametrize("value", FEED)
def test_mixed_addition_keeps_the_fraction(value, tmp_path):
    """The rule is about the domain, not about multiplication."""
    module, source, result = _binary_with_integer_constant(
        "Add", 1, constant_first=True,
    )
    produced = _run(module, source, result, value, tmp_path / "add")
    assert produced == pytest.approx(1.0 + value, rel=0, abs=0)


def test_integer_only_operation_stays_integral(tmp_path):
    """Promotion must not drag integer-only arithmetic into floating point.

    Index and count arithmetic has no float operand, so nothing promotes and
    the declared integral result is preserved exactly.
    """
    left = SSAValue(2000, dtype="int", shape=())
    right = SSAValue(2001, dtype="int", shape=())
    result = SSAValue(2002, dtype="int", shape=())
    function = Function(
        "program",
        [left],
        {
            "entry": BasicBlock("entry", [
                Instr("Const", [], right, attributes={"value": 7}),
                Instr("Mul", [left, right], result),
                Instr("Ret", [result], None),
            ])
        },
    )
    module = IRModule({"program": function})
    artifact = emit_ssa_function_to_llvm(module, "program")
    assert artifact.shortfalls == (), artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / "integral")
    execution = prepare_artifact_execution(
        native, {int(left.id): np.asarray([6], dtype=np.int64)},
    )
    execution.run()
    produced = np.asarray(execution.buffers[int(result.id)]).reshape(-1)[0]
    assert int(produced) == 42


@pytest.mark.parametrize(("gate", "expected"), [(0.0, 0), (1.0, 7)])
def test_promoted_float_register_is_reconciled_to_declared_integer_result(
    gate, expected, tmp_path,
):
    """A compiler index may be multiplied by a numeric predicate.

    The operation executes in the promoted floating domain, then converts
    back to the result cell's declared integer type before storing.  Keeping
    the declared type as the register type produced invalid LLVM of the form
    ``store i32 %double_register`` during compiler self-bootstrap.
    """

    index = SSAValue(3000, dtype="int", shape=())
    predicate = SSAValue(3001, dtype="float64", shape=())
    result = SSAValue(3002, dtype="int", shape=())
    function = Function(
        "program",
        [index, predicate],
        {
            "entry": BasicBlock("entry", [
                Instr("Mul", [index, predicate], result),
                Instr("Ret", [result], None),
            ])
        },
    )
    artifact = emit_ssa_function_to_llvm(IRModule({"program": function}), "program")
    assert artifact.shortfalls == (), artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / f"gate-{expected}")
    execution = prepare_artifact_execution(native, {
        int(index.id): np.asarray([7], dtype=np.int64),
        int(predicate.id): np.asarray([gate], dtype=np.float64),
    })
    execution.run()
    produced = np.asarray(execution.buffers[int(result.id)]).reshape(-1)[0]
    assert int(produced) == expected
