"""The direct symbolic producer retains graph identities for native emission."""
import sympy

from src.compiler.symbolic_equation_compiler import compile_sympy_equations
from src.compiler.ssa_c_backend import emit_ssa_function_to_c
from src.compiler.emission_concordance import value_cell
from src.compiler.identity_concordance import concordance_report


def test_symbolic_native_scalar_has_exact_graph_identity():
    x, y = sympy.symbols("native_identity_x native_identity_y")
    for _ in range(2):
        compiled = compile_sympy_equations(
            [sympy.Eq(y, x + 1)], name="symbolic_native_identity")
        book = compiled.module.metadata["identity_book"]
        for value in (*compiled.function.args, *(i.res for i in compiled.instructions if i.res is not None)):
            cell = value_cell(book, compiled.function, value)
            assert cell is not None
            assert book.edges_into(cell)
        emitted = emit_ssa_function_to_c(compiled.module, compiled.function.name)
        assert emitted.complete
        assert emitted.input_names == ("native_identity_x",)
        assert emitted.output_names == ("native_identity_y",)
        assert "0 finding(s)" in concordance_report(compiled.module).splitlines()[0]
