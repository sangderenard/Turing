import sympy

from src.compiler.ssa_fortran_backend import emit_module as emit_fortran
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.compiler.symbolic_equation_compiler import (
    SymbolicPublication,
    compile_sympy_equations,
)
from src.compiler.symbolic_fluid_model import (
    compile_symbolic_fluid_step,
    symbolic_viscous_shallow_water_equations,
)


def test_named_sympy_equations_share_one_process_graph_and_ssa_function():
    x, y = sympy.symbols("x y")
    sum_xy = x + y
    equations = (
        sympy.Eq(sympy.Symbol("u_next"), sum_xy * sum_xy, evaluate=False),
        sympy.Eq(sympy.Symbol("v_next"), sum_xy + 1, evaluate=False),
    )

    compiled = compile_sympy_equations(
        equations,
        publications=(
            SymbolicPublication("u_next", "fluid.velocity.x", unit="m/s"),
        ),
    )

    assert compiled.equations == equations
    assert compiled.process_graph.G.graph["symbolic_source"] == "sympy"
    assert compiled.process_graph.G.graph["sympy_translation_fallbacks"] == ()
    assert compiled.function.metadata["symbolic_source"] == "sympy"
    assert compiled.function.metadata["output_names"] == ("u_next", "v_next")
    assert compiled.input_ids.keys() == {"x", "y"}
    assert compiled.output_ids.keys() == {"u_next", "v_next"}
    # SymPy's variadic Add is normalized once, at ProcessGraph, into the exact
    # binary arity shared by every repository backend.
    add_nodes = [
        data for _node_id, data in compiled.process_graph.G.nodes(data=True)
        if data.get("op") == "Add"
    ]
    assert add_nodes
    assert all(len(data["parents"]) == 2 for data in add_nodes)
    assert all(instruction.op != "Tuple" for instruction in compiled.instructions)
    assert compiled.instructions[-1].op == "Ret"
    assert len(compiled.instructions[-1].args) == 2


def test_symbolic_equation_ssa_is_accepted_directly_by_llvm_and_fortran():
    x, y = sympy.symbols("x y")
    compiled = compile_sympy_equations((
        sympy.Eq(sympy.Symbol("result"), x + 2 * y, evaluate=False),
    ))
    outputs = compiled.function.blocks["entry"].instrs[-1].args

    llvm = emit_ssa_function_to_llvm(
        compiled.module, compiled.function.name,
    )
    fortran = emit_fortran(
        compiled.module,
        name="symbolic_equation_test",
        outputs={compiled.function.name: outputs},
    )

    assert llvm.complete, [shortfall.reason for shortfall in llvm.shortfalls]
    assert fortran.complete, [shortfall.format() for shortfall in fortran.shortfalls]
    assert "fadd double" in llvm.llvm_ir
    assert "t4 = (t0 + (2 * t2))" in fortran.source
    assert llvm.output_publications == ()
    assert fortran.api.metadata["semantic_outputs"] == ()


def test_linkable_sympy_graph_normalizes_exact_rationals_to_float64_literals():
    x = sympy.Symbol("x")
    compiled = compile_sympy_equations((
        sympy.Eq(
            sympy.Symbol("result"),
            x * sympy.Rational(1, 2),
            evaluate=False,
        ),
    ))

    literals = tuple(
        (data.get("attributes") or {}).get("value", data.get("constant"))
        for _node_id, data in compiled.process_graph.G.nodes(data=True)
        if str(data.get("type") or data.get("op") or "").casefold()
        in {"const", "constant"}
    )
    assert 0.5 in literals
    assert all(not isinstance(value, sympy.Rational) for value in literals)


def test_symbolic_equation_outputs_are_simultaneous_not_implicit_recurrence():
    x = sympy.Symbol("x")
    next_x = sympy.Symbol("x_next")

    try:
        compile_sympy_equations((
            sympy.Eq(next_x, next_x + x, evaluate=False),
        ))
    except ValueError as error:
        assert "next-state outputs" in str(error)
    else:  # pragma: no cover - the invariant must not silently weaken
        raise AssertionError("recursive symbolic output was accepted")


def test_fluid_program_is_authored_as_sympy_equations_and_lowers_without_holes():
    model = symbolic_viscous_shallow_water_equations()
    assert len(model.equations) == 11
    assert all(isinstance(equation, sympy.Equality) for equation in model.equations)
    assert model.state_outputs == (
        "height_next", "momentum_x_next", "momentum_y_next", "tracer_next",
    )

    compiled = compile_symbolic_fluid_step()
    outputs = compiled.function.blocks["entry"].instrs[-1].args
    llvm = emit_ssa_function_to_llvm(compiled.module, compiled.function.name)
    fortran = emit_fortran(
        compiled.module,
        name="symbolic_fluid_test",
        outputs={compiled.function.name: outputs},
    )

    assert compiled.process_graph.G.graph["sympy_translation_fallbacks"] == ()
    assert len(compiled.process_graph.G) > 300
    assert len(compiled.input_ids) == 30
    assert llvm.complete, [shortfall.reason for shortfall in llvm.shortfalls]
    assert fortran.complete, [shortfall.format() for shortfall in fortran.shortfalls]
    assert llvm.output_publications == fortran.api.metadata["semantic_outputs"]
    assert {row["output"] for row in llvm.output_publications} == {
        row.output for row in compiled.publications
    }


def test_fluid_state_update_is_unclamped_and_publishes_dt_rejection_metrics():
    model = symbolic_viscous_shallow_water_equations()
    by_name = {str(equation.lhs): equation.rhs for equation in model.equations}

    for name in model.state_outputs:
        assert by_name[name].func not in {sympy.Min, sympy.Max}
    assert "height_violation" in by_name
    assert "tracer_violation" in by_name
    assert "wave_speed" in by_name
    assert {
        row.semantic for row in model.publications if row.presentation == "metric"
    } == {
        "dt.metric.wave_speed",
        "dt.error.height_positivity",
        "dt.error.tracer_bounds",
    }


def test_tanh_lowers_to_llvm_as_a_declared_libm_call():
    """tanh has no LLVM intrinsic; the vehicle body uses it 21 times.

    Before this entry the whole vehicle body failed LLVM emission (21
    'no likeness-table entry' shortfalls and ~700 operands unavailable
    downstream) while C and Fortran emitted it.
    """

    x = sympy.Symbol("tanh_x")
    compiled = compile_sympy_equations(
        (sympy.Eq(sympy.Symbol("tanh_y"), sympy.tanh(x) * 2, evaluate=False),),
        name="tanh_probe",
    )
    artifact = emit_ssa_function_to_llvm(compiled.module, compiled.function.name)
    assert artifact.complete, [s.reason for s in artifact.shortfalls]
    assert "call double @tanh(double" in artifact.llvm_ir
    assert "declare double @tanh(double)" in artifact.llvm_ir
    # Fortran had the same gap: its SSA-op table lacked "Tanh" (only the
    # recorded-tape "tanh" key existed).
    ret = next(i for b in compiled.function.blocks.values() for i in b.instrs if i.op == "Ret")
    fortran = emit_fortran(
        compiled.module, name="tanh_probe_fortran",
        outputs={compiled.function.name: tuple(ret.args)}, progress=lambda _m: None,
    )
    assert fortran.complete, [s.format() for s in fortran.shortfalls]
    assert "tanh(" in fortran.source
