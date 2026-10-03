"""Nonfinite symbolic literals remain constants through native lowering."""

import numpy as np
import pytest
import sympy as sp

from src.compiler.identity_concordance import concordance_report
from src.compiler.native_package import piece_from_law
from src.compiler.symbolic_equation_compiler import compile_sympy_equations


@pytest.mark.parametrize("infinity", (sp.oo, -sp.oo))
def test_symbolic_infinity_equality_uses_the_authored_constant(tmp_path, infinity):
    slew = sp.Symbol("slew")
    expression = sp.Piecewise((sp.Integer(1), sp.Eq(slew, infinity)),
                             (sp.Integer(0), True))
    law = compile_sympy_equations(
        (sp.Eq(sp.Symbol("result"), expression, evaluate=False),),
        name="symbolic_infinity_equality",
    )
    piece = piece_from_law(law, "symbolic_infinity_equality", 4,
                           directory=tmp_path)
    print("authored source:\n" + piece.source, flush=True)
    print("\n".join(concordance_report(piece.module).splitlines()[:4]), flush=True)
    samples = np.array([np.inf, -np.inf, 0.0, 1.0])
    result, = piece(samples)
    expected = np.array([float(expression.subs(slew, sp.sympify(value)))
                         for value in samples])
    print(f"native={result!r} symbolic={expected!r}", flush=True)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("infinity", (sp.oo, -sp.oo))
def test_symbolic_infinity_is_published_by_the_native_piece(tmp_path, infinity):
    command = sp.Symbol("command")
    expression = sp.Piecewise((infinity, command > 0), (sp.Integer(0), True))
    law = compile_sympy_equations(
        (sp.Eq(sp.Symbol("result"), expression, evaluate=False),),
        name="symbolic_infinity_publication",
    )
    piece = piece_from_law(law, "symbolic_infinity_publication", 4,
                           directory=tmp_path)
    assert "result" in piece.output_ids
    result, = piece(np.array([-1.0, 0.0, 1.0, np.inf]))
    np.testing.assert_array_equal(result, np.array([0.0, 0.0, float(infinity),
                                                  float(infinity)]))
    print("direct infinity publication:", result, flush=True)
