"""A SymPy integer constant is a number where its consumer reads a value and
an integer where its consumer's declared parameter is one.

aa5f1aac made plain-int constants float64 (a Piecewise arm ``1`` stored as
``i64 1`` read back as 5e-324).  That also floatified the quadrature's axis
constants (``unsqueeze(-1.0)``), and the Integral's AbstractTensor stage
raised.  The consuming operation's declared ``AbstractTensor`` signature
decides (``symbolic_equation_compiler._structural_constants``).

    python -m pytest tests/test_symbolic_structural_constants.py -q
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from src.compiler.symbolic_equation_compiler import compile_sympy_equations


def _constants(compilation):
    consumers: dict[int, list] = {}
    for instruction in compilation.instructions:
        for position, argument in enumerate(instruction.args):
            consumers.setdefault(int(argument.id), []).append((instruction.op, position))
    return [
        (instruction.attributes.get("constant"), consumers.get(int(instruction.res.id), []))
        for instruction in compilation.instructions if instruction.op == "Const"]


def test_piecewise_arm_constants_stay_numbers():
    x = sp.Symbol("x", real=True)
    law = [sp.Eq(sp.Symbol("y"), sp.Piecewise((1, x > 0), (0, True)), evaluate=False)]
    compilation = compile_sympy_equations(law, name="structural_piecewise_arm")
    arms = [payload for payload, uses in _constants(compilation)
            if any(op in {"Select", "select"} for op, _ in uses)]
    assert arms and all(type(payload) is float for payload in arms), arms


def test_quadrature_axis_constants_stay_integers_and_the_stage_runs():
    from src.compiler.vehicle_python_compilation import symbolic_abstract_tensor_source
    from src.common.tensors import AbstractTensor

    s, L = sp.Symbol("s", real=True), sp.Symbol("L", real=True)
    F = sp.Function("F")
    law = [sp.Eq(sp.Symbol("cost"), sp.Integral(sp.sqrt(F(s) ** 2), (s, 0, L)),
                 evaluate=False)]
    name = "structural_quadrature_axis"
    compilation = compile_sympy_equations(law, name=name)
    structural = [(payload, uses) for payload, uses in _constants(compilation)
                  if any(op in {"unsqueeze", "sum"} and position == 1
                         for op, position in uses)]
    assert structural
    for payload, _uses in structural:
        assert type(payload) is int, structural

    namespace = {"AbstractTensor": AbstractTensor,
                 "F": lambda t: (t * 0.5).cos() + 2.0}
    exec(symbolic_abstract_tensor_source(compilation, name), namespace)
    lengths = np.array([0.5, 1.0, 2.0, 3.0])
    got = np.asarray(namespace[name](AbstractTensor.get_tensor(lengths)).data)
    nodes, weights = np.polynomial.legendre.leggauss(5)
    want = np.array([
        (length / 2) * np.sum(weights * (np.cos(0.5 * (length / 2) * (nodes + 1)) + 2.0))
        for length in lengths])
    assert np.allclose(got, want, rtol=1e-13, atol=0), (got, want)
