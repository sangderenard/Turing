import re

import pytest

from src.compiler.llvm_optimizing_pipeline import (
    OptimizationProfile,
    REFERENCE_PROFILE,
    annotate_pointer_parameters,
    apply_fast_math,
    compare_profiles,
    optimize_ir,
)
from src.compiler.ssa_fortran_backend import (
    compile_module,
    emit_function,
    emit_module,
    fortran_compiler,
)
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSAValue,
)


ELEMENTWISE_IR = """
define void @axpy(ptr %a, ptr %b, ptr %out, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %done = icmp sge i32 %i, %n
  br i1 %done, label %exit, label %body
body:
  %i64 = sext i32 %i to i64
  %ap = getelementptr inbounds double, ptr %a, i64 %i64
  %bp = getelementptr inbounds double, ptr %b, i64 %i64
  %op = getelementptr inbounds double, ptr %out, i64 %i64
  %av = load double, ptr %ap, align 8
  %bv = load double, ptr %bp, align 8
  %m = fmul double %av, 2.500000e+00
  %s = fadd double %m, %bv
  store double %s, ptr %op, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop
exit:
  ret void
}
"""


def _elementwise_function():
    a = SSAValue(0, "float64", (64,))
    b = SSAValue(1, "float64", (64,))
    scaled = SSAValue(3, "float64", (64,))
    summed = SSAValue(4, "float64", (64,))
    activated = SSAValue(5, "float64", (64,))
    total = SSAValue(2, "float64")
    block = BasicBlock(
        "entry",
        [
            Instr("mul", [a], scaled, attributes={"right_scalar": 2.5}),
            Instr("add", [scaled, b], summed),
            Instr("tanh", [summed], activated),
            Instr("sum", [activated], total),
            Instr("Ret", [], SSAValue(9)),
        ],
    )
    function = Function("axpy_tanh_sum", [a, b], {"entry": block})
    return function, activated, total


def _loop_function():
    x = SSAValue(0, "float64")
    n = SSAValue(1, "float64")
    seed = SSAValue(2, "float64")
    acc = SSAValue(3, "float64")
    nxt = SSAValue(4, "float64")
    cond = SSAValue(5, "bool")
    blocks = {
        "entry": BasicBlock(
            "entry",
            [
                Instr("const", [], seed, attributes={"constant": 0.0}),
                Instr("Br", [], SSAValue(90), attributes={"target": "loop"}),
            ],
        ),
        "loop": BasicBlock(
            "loop",
            [
                Instr(
                    "Phi",
                    [],
                    acc,
                    attributes={"incoming": [("entry", seed), ("body", nxt)]},
                ),
                Instr("Lt", [acc, n], cond),
                Instr(
                    "CondBr",
                    [cond],
                    SSAValue(91),
                    attributes={"true": "body", "false": "exit"},
                ),
            ],
        ),
        "body": BasicBlock(
            "body",
            [
                Instr("add", [acc, x], nxt),
                Instr("Br", [], SSAValue(92), attributes={"target": "loop"}),
            ],
        ),
        "exit": BasicBlock("exit", [Instr("Ret", [], SSAValue(93))]),
    }
    return Function("accumulate", [x, n], blocks), acc


# ---------------------------------------------------------------- Fortran


def test_elementwise_ssa_emits_whole_array_fortran():
    function, activated, total = _elementwise_function()
    module = emit_module(
        IRModule({"axpy_tanh_sum": function}),
        name="turing_demo",
        outputs={"axpy_tanh_sum": [activated, total]},
    )
    source = module.source

    assert module.complete, [s.format() for s in module.shortfalls]
    assert 'bind(C, name="axpy_tanh_sum")' in source
    assert "use, intrinsic :: iso_c_binding" in source
    # Whole-array expressions rather than explicit element loops, and the
    # single-use chain folded into one statement.
    assert "t5 = tanh(((t0 * 2.5_c_double) + t1))" in source
    assert "t2 = sum(t5)" in source


def test_single_use_temporaries_are_fused_into_one_expression():
    """One SSA chain must become one array statement, not one per step.

    A statement per SSA step materialises an array temporary per step, turning
    a single fused traversal into N passes over memory.  At array sizes past
    cache that dominates the runtime, so this is a correctness-of-intent test
    for the emitter, not a cosmetic one.
    """

    function, activated, total = _elementwise_function()
    source = emit_function(function, outputs=[activated, total]).source

    # The intermediate values must not appear as declared arrays at all.
    assert "t3(" not in source
    assert "t4(" not in source
    # Exactly two assignments survive: the fused elementwise chain, and the
    # reduction that genuinely cannot fuse into it.
    assignments = re.findall(r"^\s*t\d+ = ", source, flags=re.MULTILINE)
    assert len(assignments) == 2, source


def test_bind_c_arrays_are_explicit_shape_over_a_passed_extent():
    """A bind(C) procedure may not take assumed-shape dummies.

    ``x(:)`` needs a descriptor a C caller cannot construct before Fortran 2018
    / TS 29113, and gfortran rejects it outright.  Arrays must be explicit-shape
    over an extent supplied by the caller.
    """

    function, activated, total = _elementwise_function()
    source = emit_function(function, outputs=[activated, total]).source

    assert "integer(c_int), intent(in), value :: n_elements" in source
    assert "intent(in) :: t0(n_elements)" in source
    assert "intent(out) :: t5(n_elements)" in source
    assert "intent(out) :: t2" in source
    # No assumed-shape dummy may survive anywhere in a bind(C) subroutine,
    # and nothing may reach the heap.
    assert "(:)" not in source
    assert "allocatable" not in source


def test_unreferenced_block_labels_are_not_emitted():
    """A label nothing branches to is a warning and pure noise."""

    function, activated, total = _elementwise_function()
    straight_line = emit_function(function, outputs=[activated, total]).source
    assert "continue" not in straight_line

    loop, acc = _loop_function()
    with_branches = emit_function(loop, outputs=[acc]).source
    assert "continue" in with_branches


def test_control_flow_lowers_phi_into_predecessor_assignments():
    function, acc = _loop_function()
    source = emit_function(function, outputs=[acc]).source

    # Phi is realised by assigning in each predecessor before the branch,
    # never by a Fortran construct pretending to be a phi.
    entry_section = source.split("! block loop")[0]
    body_section = source.split("! block body")[1]
    assert "t3 = t2" in entry_section
    assert "t3 = t4" in body_section
    # The comparison feeding the branch is single-use, so it folds into the
    # condition rather than occupying a named temporary.
    assert "if ((t3 < t1)) then" in source
    assert source.count("goto") >= 3


def test_values_live_across_blocks_are_not_fused_away():
    """Inlining must never move a computation across control flow."""

    function, acc = _loop_function()
    source = emit_function(function, outputs=[acc]).source

    # t4 is defined in 'body' and consumed by the loop-header phi, so it must
    # remain a real assignment inside the block that computes it.
    assert "t4 = (t3 + t0)" in source
    assert "real(c_double) :: t4" in source


def test_unsupported_operation_is_reported_not_guessed():
    value = SSAValue(0, "float64", (8,))
    result = SSAValue(1, "float64", (8,))
    block = BasicBlock("entry", [Instr("einsum", [value], result)])
    function = Function("mystery", [value], {"entry": block})

    subroutine = emit_function(function, outputs=[result])
    assert not subroutine.complete
    assert subroutine.shortfalls[0].op == "einsum"
    assert "UNSUPPORTED einsum" in subroutine.source


@pytest.mark.skipif(
    fortran_compiler() is None, reason="no Fortran compiler installed"
)
def test_generated_fortran_actually_compiles(tmp_path):
    function, activated, total = _elementwise_function()
    module = emit_module(
        IRModule({"axpy_tanh_sum": function}),
        name="turing_demo",
        outputs={"axpy_tanh_sum": [activated, total]},
    )
    library = compile_module(module, directory=tmp_path)
    assert library.exists()


# ------------------------------------------------------------------- LLVM


def test_reference_profile_reproduces_the_unoptimized_path():
    result = optimize_ir(ELEMENTWISE_IR, REFERENCE_PROFILE)
    assert result.profile.opt == 0
    assert not result.profile.annotate_noalias
    # Scalar double arithmetic only: no packed SIMD.
    assert result.vector_instruction_count == 0


def test_optimizing_profile_produces_packed_simd():
    reference, optimized = compare_profiles(ELEMENTWISE_IR)

    assert reference.vector_instruction_count == 0
    assert optimized.vectorized
    # Packed forms, not the scalar sd/ss forms that also use xmm registers.
    assert re.search(r"\bv?(add|mul)pd\b", optimized.assembly)


def test_noalias_removes_runtime_alias_versioning_not_vectorization():
    """``noalias`` buys smaller code, not the ability to vectorize.

    LLVM's loop vectorizer handles unproven aliasing by versioning the loop
    behind a runtime memory check, so it vectorizes with or without the
    attribute.  Asserting that ``noalias`` *enables* SIMD would be wrong.  What
    it actually removes is the check and the scalar fallback copy — which is
    the advantage Fortran gets for free from its argument rules.
    """

    without = optimize_ir(
        ELEMENTWISE_IR,
        OptimizationProfile(annotate_noalias=False),
    )
    with_noalias = optimize_ir(
        ELEMENTWISE_IR,
        OptimizationProfile(annotate_noalias=True),
    )

    assert without.vectorized and with_noalias.vectorized
    assert with_noalias.vector_instruction_count == (
        without.vector_instruction_count
    )
    # The saving is the elided versioning path.
    assert len(with_noalias.assembly.splitlines()) < len(
        without.assembly.splitlines()
    )


def test_pointer_annotation_is_confined_to_pointer_parameters():
    annotated = annotate_pointer_parameters(ELEMENTWISE_IR)
    assert "ptr noalias %a" in annotated
    assert "ptr noalias %out" in annotated
    # The integer parameter must not be touched.
    assert "i32 %n" in annotated
    assert "i32 noalias" not in annotated
    # Applying twice must not double-annotate.
    assert annotate_pointer_parameters(annotated) == annotated


def test_fast_math_is_off_by_default_and_explicit_when_requested():
    assert not OptimizationProfile().fast_math
    assert "fast" not in optimize_ir(ELEMENTWISE_IR).hardened_ir.split("fadd")[0]

    fast = apply_fast_math(ELEMENTWISE_IR)
    assert "fmul fast" in fast
    assert "fadd fast" in fast
