import math

import pytest

from src.compiler.bounded_constants import PiSolver, materialize_pi
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


def test_machin_pi_has_a_tunable_explicit_error_contract():
    coarse = materialize_pi("machin", 1.0e-6)
    fine = materialize_pi("machin", 1.0e-12)

    assert coarse.absolute_error_bound <= 1.0e-6
    assert fine.absolute_error_bound <= 1.0e-12
    assert abs(fine.value - math.pi) <= fine.absolute_error_bound
    assert fine.terms_atan_1_5 > coarse.terms_atan_1_5
    assert fine.llvm_symbol == "turing_machin_pi_f64"


def test_repo_ssa_pi_lowers_to_selected_bounded_llvm_implementation():
    value = SSAValue(0, "float64")
    function = Function("pi_value", [], {
        "entry": BasicBlock("entry", [
            Instr("Pi", [], value),
            Instr("Ret", [value], None),
        ]),
    })

    artifact = emit_ssa_function_to_llvm(
        IRModule({function.name: function}),
        function.name,
        pi_solver="machin",
        pi_epsilon=1.0e-10,
    )

    assert artifact.shortfalls == ()
    assert "define internal double @turing_machin_pi_f64()" in artifact.llvm_ir
    assert "call double @turing_machin_pi_f64()" in artifact.llvm_ir


def test_repo_ssa_pi_can_be_rejected_by_backend_policy():
    value = SSAValue(0, "float64")
    function = Function("pi_value", [], {
        "entry": BasicBlock("entry", [Instr("Pi", [], value)]),
    })

    artifact = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name, pi_solver="reject",
    )

    assert not artifact.complete
    assert artifact.shortfalls[0].operation == "Pi"


def test_literal_pi_reports_f64_representation_bound():
    literal = materialize_pi(PiSolver.LITERAL)

    assert literal.value == math.pi
    assert literal.absolute_error_bound == math.ulp(math.pi) * 0.5


def test_invalid_machin_pi_epsilon_is_rejected():
    with pytest.raises(ValueError, match="pi epsilon"):
        materialize_pi("machin", 1.0)
