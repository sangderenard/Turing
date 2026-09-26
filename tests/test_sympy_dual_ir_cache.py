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



def _write_authored_program(path, *, revision: str) -> None:
    path.write_text(
        "import sympy\n"
        "CALLS = []\n"
        "\n"
        "def produce():\n"
        "    CALLS.append(1)\n"
        "    x = sympy.Symbol('cached_program_x')\n"
        "    equation = sympy.Eq(sympy.Symbol('cached_program_y'), x * x + 1, evaluate=False)\n"
        "    return (equation,), {'x': x}\n"
        f"# revision {revision}\n",
        encoding="utf-8",
    )


def _load_authored_program(path, name="authored_symbolic_program_fixture"):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_symbolic_program_is_built_once_per_source_revision(tmp_path, monkeypatch):
    """The finished compilation loads without running the authored producer.

    The producer's sympy construction is the cost this cache exists to avoid,
    so a hit must not call it at all, and the sibling that needs only the
    equations must share the same stored program.
    """

    from src.compiler.symbolic_equation_compiler import (
        compile_symbolic_program, symbolic_equations_cached,
    )

    monkeypatch.setenv("TURING_SYMPY_DUAL_IR_CACHE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "authored_symbolic_program.py"
    _write_authored_program(path, revision="one")
    module = _load_authored_program(path)

    first = compile_symbolic_program(module.produce, name="cached_program")
    assert first.cache_hit is False
    assert module.CALLS == [1]

    module.CALLS.clear()
    second = compile_symbolic_program(module.produce, name="cached_program")
    assert second.cache_hit is True
    assert module.CALLS == []
    assert second.function.metadata["argument_names"] == first.function.metadata["argument_names"]
    assert second.function.metadata["output_names"] == ("cached_program_y",)

    equations, symbols = symbolic_equations_cached(module.produce)
    assert module.CALLS == []
    assert [str(equation.lhs) for equation in equations] == ["cached_program_y"]
    assert str(symbols["x"]) == "cached_program_x"


def test_symbolic_program_cache_rebuilds_when_the_source_file_changes(tmp_path, monkeypatch):
    """An edit to the producer's file must never be served a stale program."""

    from src.compiler.symbolic_equation_compiler import compile_symbolic_program

    monkeypatch.setenv("TURING_SYMPY_DUAL_IR_CACHE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "authored_symbolic_program.py"
    _write_authored_program(path, revision="one")
    module = _load_authored_program(path, "authored_symbolic_program_revised")
    first = compile_symbolic_program(module.produce, name="cached_program")
    assert first.cache_hit is False

    _write_authored_program(path, revision="two")
    # Re-execute the edited file under the SAME module name: the producer's
    # identity is unchanged, only its source bytes differ.
    module = _load_authored_program(path, "authored_symbolic_program_revised")
    assert module.CALLS == []
    rebuilt = compile_symbolic_program(module.produce, name="cached_program")
    assert rebuilt.cache_hit is False
    assert module.CALLS == [1]
    assert rebuilt.cache_identity != first.cache_identity
