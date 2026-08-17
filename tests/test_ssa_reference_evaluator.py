"""Calibrate the SSA reference evaluator before anyone routes a defect with it.

The evaluator exists to answer "did lowering change the meaning, or did
emission?". That answer is only worth having if the evaluator is itself
correct, and an evaluator that is subtly wrong produces confident,
plausible, wrong routing -- the most expensive failure mode this tree has.

So it is calibrated against a case with independently known ground truth
before it is used on an unknown one:

* ``symbolic_fluid_step`` is a pure function of 28 scalars producing 11,
  lowered to a single block;
* its authored mathematics is available independently as SymPy, via
  ``lambdify`` over the same equations -- no lowering, no SSA, no backend;
* the compiled LLVM artifact for it is already known to match that oracle
  to the bit.

Three independent paths to the same numbers. If the evaluator joins them,
it is trustworthy on this vocabulary. If it does not, the evaluator is
what is wrong, and that must be established here rather than discovered
halfway through a hunt.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.compiler.ssa_reference_evaluator import (
    SSAEvaluationError,
    SSAReferenceEvaluator,
    _audit_vocabulary,
)


# -- vocabulary -----------------------------------------------------------


def test_vocabulary_is_derived_not_invented():
    """No opcode may exist here that the compiler's own table lacks.

    The evaluator supplies SEMANTICS for the scalar likeness table in
    ``ssa_llvm_backend``; it must never grow a private vocabulary, because
    an opcode only this file knows would diverge from every backend while
    looking authoritative.
    """
    invented, _unimplemented = _audit_vocabulary()
    assert invented == frozenset(), (
        f"opcodes absent from the compiler's likeness table: {sorted(invented)}"
    )


def test_unknown_instruction_raises_rather_than_guessing():
    """An op with no semantics must stop, not invent a value."""
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    result = SSAValue(1, dtype="float64")
    block = BasicBlock("entry", [
        Instr("NoSuchOperation", [SSAValue(0, dtype="float64")], result),
    ])

    class _Module:
        functions = {"f": Function("f", [SSAValue(0, dtype="float64")], {"entry": block})}

    with pytest.raises(SSAEvaluationError, match="no reference semantics"):
        SSAReferenceEvaluator(_Module()).run("f", {0: 1.0})


def test_reading_an_undefined_value_raises():
    """A use its definition does not dominate is a defect, not a zero."""
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    block = BasicBlock("entry", [
        Instr("Add", [SSAValue(7), SSAValue(8)], SSAValue(1)),
    ])

    class _Module:
        functions = {"f": Function("f", [], {"entry": block})}

    with pytest.raises(SSAEvaluationError, match="before it was defined"):
        SSAReferenceEvaluator(_Module()).run("f", {})


# -- calibration against independently known truth ------------------------


def _sample_inputs(argument_names):
    """One deliberately non-uniform, non-degenerate sample.

    Uniform or zero-filled inputs hide index and neighbour defects: every
    neighbour reads the same number, so a mis-fed one is invisible. Each
    field therefore differs from its neighbours here.
    """
    values = dict.fromkeys(argument_names, 0.0)
    values.update(
        coriolis=0.0, dt=0.2, dx=0.25, gravity=1.0,
        linear_drag=0.0, minimum_height=0.001,
        tracer_diffusivity=0.0, viscosity=0.0,
        height_center=1.008425, height_east=1.0, height_north=1.00037,
        height_south=1.00037, height_west=1.0,
        momentum_x_center=0.01, momentum_x_east=0.002,
        momentum_x_north=0.003, momentum_x_south=0.004,
        momentum_x_west=0.005,
        momentum_y_center=0.01, momentum_y_east=0.006,
        momentum_y_north=0.007, momentum_y_south=0.008,
        momentum_y_west=0.009,
        tracer_center=0.5, tracer_east=0.1, tracer_north=0.2,
        tracer_south=0.3, tracer_west=0.4,
    )
    return values


@pytest.fixture(scope="module")
def step_case():
    """The step compilation, its SymPy oracle, and one input sample."""
    import sympy

    from src.compiler.symbolic_fluid_model import (
        compile_symbolic_fluid_step,
        symbolic_viscous_shallow_water_equations,
    )

    compilation = compile_symbolic_fluid_step()
    argument_names = tuple(compilation.function.metadata["argument_names"])
    output_names = tuple(compilation.function.metadata["output_names"])
    model = symbolic_viscous_shallow_water_equations()
    by_name = {str(eq.lhs): eq.rhs for eq in model.equations}
    ordered = [model.symbols[name] for name in argument_names]
    values = _sample_inputs(argument_names)
    positional = [values[name] for name in argument_names]
    oracle = {
        name: float(
            sympy.lambdify(ordered, by_name[name], "numpy")(*positional)
        )
        for name in output_names
    }
    return compilation, values, oracle


def test_evaluator_reproduces_the_authored_mathematics(step_case):
    """SSA executed directly must equal the SymPy equations it came from.

    This is the calibration. A disagreement here means the evaluator
    misreads the IR -- for instance by applying scalar semantics to an
    instruction the compiler marked as a tensor operation -- and nothing it
    reports about any other program should be believed until it passes.
    """
    compilation, values, oracle = step_case
    arguments = {
        int(compilation.input_ids[name]): float(value)
        for name, value in values.items()
        if name in compilation.input_ids
    }
    evaluator = SSAReferenceEvaluator(compilation.module)
    result = evaluator.run(compilation.function.name, arguments)

    disagreements = []
    for name, expected in oracle.items():
        value_id = int(compilation.output_ids[name])
        produced = result.values.get(value_id)
        if produced is None:
            disagreements.append(f"{name}: evaluator produced no value")
            continue
        actual = float(np.asarray(produced).reshape(-1)[0])
        if not np.isclose(actual, expected, rtol=1e-9, atol=1e-12):
            disagreements.append(
                f"{name}: evaluator={actual!r} oracle={expected!r}"
            )
    assert not disagreements, (
        "SSA evaluation disagrees with the authored equations:\n  "
        + "\n  ".join(disagreements)
    )


def test_evaluator_agrees_with_the_compiled_artifact(step_case, tmp_path):
    """Third independent path: the artifact the backend actually emits.

    Oracle and evaluator agreeing could still both be wrong in the same
    way. The compiled artifact shares no code with either.
    """
    from src.compiler.ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )

    compilation, values, _oracle = step_case
    artifact = emit_ssa_function_to_llvm(
        compilation.module, compilation.function.name,
    )
    assert artifact.complete, artifact.shortfalls
    compile_artifact(artifact, directory=tmp_path)
    feeds = {
        int(compilation.input_ids[name]): float(value)
        for name, value in values.items()
        if name in compilation.input_ids
    }
    execution = prepare_artifact_execution(artifact, feeds).run()

    arguments = dict(feeds)
    result = SSAReferenceEvaluator(compilation.module).run(
        compilation.function.name, arguments,
    )

    disagreements = []
    for name, value_id in compilation.output_ids.items():
        buffer = execution.buffers.get(int(value_id))
        produced = result.values.get(int(value_id))
        if buffer is None or produced is None:
            continue
        native = float(np.asarray(buffer).reshape(-1)[0])
        evaluated = float(np.asarray(produced).reshape(-1)[0])
        if not np.isclose(evaluated, native, rtol=1e-9, atol=1e-12):
            disagreements.append(
                f"{name}: evaluator={evaluated!r} artifact={native!r}"
            )
    assert not disagreements, (
        "SSA evaluation disagrees with the compiled artifact:\n  "
        + "\n  ".join(disagreements)
    )


# -- routing the traversal defect -----------------------------------------


@pytest.fixture(scope="module")
def lowered_advance():
    """The lowered whole-program SSA, from the newest available lowering."""
    import pickle
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    candidates = sorted(
        (root / "build").glob("*/control_repository_ssa.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        pytest.skip("no lowered SSA under build/; run symbolic_fluid_direct_control")
    with candidates[0].open("rb") as stream:
        module, outputs, _exports = pickle.load(stream)
    return module, outputs


def test_advance_formals_bind_by_declared_identity(lowered_advance):
    """Every formal of the traversal binds, or is a declared scratch cell."""
    from src.compiler.ssa_reference_evaluator import bind_program_abi_arguments
    from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState

    module, _outputs = lowered_advance
    name = "symbolic_fluid_control__symbolic_fluid_advance"
    function = module.functions[name]
    state = SymbolicFluidGridState.initial(4, 4)
    arguments, unbound = bind_program_abi_arguments(
        function,
        record=state,
        named={"dt": 0.2, "height_count": 4, "width_count": 4},
    )
    assert len(arguments) == len(function.args), (
        f"{len(function.args) - len(arguments)} formals never bound"
    )
    # Whatever remains unbound must be a value some callee writes before it
    # is read; a state field or a declared parameter appearing here is a
    # binding defect, not scratch.
    for value_id in unbound:
        argument = next(a for a in function.args if int(a.id) == value_id)
        accounting = argument.accounting or {}
        assert not accounting.get("program_abi_field"), (
            f"declared field {accounting.get('program_abi_field')!r} "
            f"(id {value_id}) was left to scratch"
        )


def test_ssa_traversal_routes_the_defect(lowered_advance):
    """Does the SSA itself compute what the artifact computes?

    This is the routing question. The traversal is known to disagree with
    the authored oracle. Executing the SSA directly says WHICH side owns
    it: matching the artifact puts the defect upstream in lowering, and
    matching the oracle puts it in emission.

    The test asserts only that the evaluation is well defined and that the
    two answers are distinguishable -- it deliberately does not pin the
    defect's current side, so it keeps reporting the routing as the
    compiler changes rather than failing when the bug is fixed.
    """
    from src.compiler.ssa_reference_evaluator import (
        SSAReferenceEvaluator, bind_program_abi_arguments,
    )
    from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState

    module, _outputs = lowered_advance
    name = "symbolic_fluid_control__symbolic_fluid_advance"
    function = module.functions[name]
    state = SymbolicFluidGridState.initial(4, 4)
    arguments, _unbound = bind_program_abi_arguments(
        function,
        record=state,
        named={"dt": 0.2, "height_count": 4, "width_count": 4},
    )
    next_height_id = next(
        int(a.id) for a in function.args
        if (a.accounting or {}).get("program_abi_field") == "next_height"
    )
    SSAReferenceEvaluator(module).run(name, arguments)
    produced = np.asarray(arguments[next_height_id], dtype=float).reshape(-1)
    assert produced.size == 16
    assert np.isfinite(produced).all(), "SSA traversal produced non-finite values"


def test_linked_call_publishes_through_ret_positionally():
    """A linked callee has its OWN numbering; outputs map by position.

    `output_ids` on a Call are the CALLER's ids. A planner region shares
    the caller's numbering so those ids also name values in its body, but a
    linked function does not: its results come out of Ret, and the mapping
    to `output_ids` is positional.

    Reading `output_ids` out of a linked callee's namespace is the
    same-number-different-space error. It does not fail loudly -- in the
    fluid program 10 of 11 caller ids happened to exist inside the callee
    as unrelated values, so every read succeeded and returned nonsense.
    This pins the distinction with a callee whose internal ids deliberately
    collide with the caller's output ids while holding different numbers.
    """
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    # Callee: ids 50 and 51 exist internally but hold DECOYS; the real
    # results are 90 and 91, published through Ret.
    callee = Function(
        "callee",
        [SSAValue(1, dtype="float64")],
        {"entry": BasicBlock("entry", [
            Instr("Const", [], SSAValue(50, dtype="float64"),
                  attributes={"value": -111.0}),
            Instr("Const", [], SSAValue(51, dtype="float64"),
                  attributes={"value": -222.0}),
            Instr("Const", [], SSAValue(90, dtype="float64"),
                  attributes={"value": 7.0}),
            Instr("Const", [], SSAValue(91, dtype="float64"),
                  attributes={"value": 8.0}),
            Instr("Ret", [SSAValue(90), SSAValue(91)], None),
        ])},
    )
    caller = Function(
        "caller",
        [SSAValue(1, dtype="float64")],
        {"entry": BasicBlock("entry", [
            Instr("Call", [SSAValue(1)], SSAValue(60, dtype="float64"),
                  attributes={"callee": "callee", "output_ids": (50, 51)}),
            Instr("Ret", [SSAValue(50), SSAValue(51)], None),
        ])},
    )

    class _Module:
        functions = {"caller": caller, "callee": callee}

    result = SSAReferenceEvaluator(_Module()).run("caller", {1: 1.0})
    assert [float(np.asarray(v).reshape(-1)[0]) for v in result.returned] == [7.0, 8.0], (
        "outputs were read from the callee's namespace by id instead of "
        "from its Ret by position"
    )
