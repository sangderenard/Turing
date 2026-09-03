import sympy

from src.compiler.sympy_dual_ir_cache import SympyDualIRCache
from src.compiler.symbolic_equation_compiler import compile_sympy_equations


def test_solved_equation_program_is_persistent_and_content_addressed(tmp_path):
    equation = sympy.Eq(sympy.Symbol("y"), sympy.Symbol("x") + 1, evaluate=False)
    record = {
        "equations": (sympy.srepr(equation),),
        "unknowns": ("y",),
        "solver": "fixture-linear-solve-v1",
    }
    calls = 0

    def solve():
        nonlocal calls
        calls += 1
        return (equation,)

    first = SympyDualIRCache("fixture-implementation-v1", root=tmp_path)
    second = SympyDualIRCache("fixture-implementation-v1", root=tmp_path)
    miss = first.solved_equations(record, solve)
    hit = second.solved_equations(record, solve)

    assert calls == 1
    assert miss.hit is False
    assert hit.hit is True
    assert hit.identity == miss.identity
    assert hit.value == (equation,)


def test_sympy_repository_dual_ir_is_reused_without_relowering(tmp_path, monkeypatch):
    monkeypatch.setenv("TURING_SYMPY_DUAL_IR_CACHE_DIR", str(tmp_path))
    x = sympy.Symbol("persistent_cache_x")
    equations = (
        sympy.Eq(
            sympy.Symbol("persistent_cache_y"), x * x + 1, evaluate=False,
        ),
    )

    first = compile_sympy_equations(equations, name="persistent_cache_probe")
    second = compile_sympy_equations(equations, name="persistent_cache_probe")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.cache_identity == first.cache_identity
    assert second.function.name == "persistent_cache_probe"
    assert second.function.metadata["symbolic_equations"] == tuple(
        sympy.srepr(equation) for equation in equations
    )


def test_runtime_parameter_values_are_not_part_of_symbolic_cache_identity(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("TURING_SYMPY_DUAL_IR_CACHE_DIR", str(tmp_path))
    parameter = sympy.Symbol("live_stiffness")
    displacement = sympy.Symbol("live_displacement")
    force = sympy.Symbol("force")
    equations = (
        sympy.Eq(force, parameter * displacement, evaluate=False),
    )

    first = compile_sympy_equations(equations, name="live_parameter_probe")
    second = compile_sympy_equations(equations, name="live_parameter_probe")

    assert first.cache_identity == second.cache_identity
    assert second.cache_hit is True
    assert set(second.input_ids) == {"live_stiffness", "live_displacement"}

