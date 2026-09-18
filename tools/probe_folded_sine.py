"""Compile folded_sine alone, at a stated width, and score it on mpmath.

The field demo's per-cell expression is imported, not restated, so what
this measures is the same mathematics the demo runs. Points are chosen to
cover every quadrant at both signs and across magnitudes, because the
defects found here were all quadrant- or sign-specific and a sample that
happened to miss one reported success.
"""
import ast
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import mpmath
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.demo_kuramoto_field import (  # noqa: E402
    core_terms, folded_sine, symbolise_numbers,
)
from tools.demo_kuramoto_field_compiled import (  # noqa: E402
    deploy, parameter_ids, prepare,
)
from src.common.tensors.signal_symbolic import (  # noqa: E402
    constant_rational, limb_decomposition,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.ssa_python_materializer import (  # noqa: E402
    materialize_function_body,
)
from src.compiler.symbolic_equation_compiler import (  # noqa: E402
    compile_sympy_equations,
)


class _IndexX(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "x" and isinstance(node.ctx, ast.Load):
            return ast.Subscript(
                value=ast.Name(id="x", ctx=ast.Load()),
                slice=ast.Name(id="i", ctx=ast.Load()), ctx=ast.Load(),
            )
        return node


def build(width: int, digits: int, backend: str = "c",
          element: str | None = None):
    sine = list(core_terms("sin", digits))
    cosine = list(core_terms("cos", digits))
    terms = max(len(sine), len(cosine))
    sine += [Fraction(0)] * (terms - len(sine))
    cosine += [Fraction(0)] * (terms - len(cosine))

    x = sympy.Symbol("x")
    quarter, neg_quarter, inv_quarter = sympy.symbols(
        "quarter neg_quarter inv_quarter"
    )
    expression = folded_sine(
        x, quarter, neg_quarter, inv_quarter,
        sympy.symbols(f"c0:{terms}"), sympy.symbols(f"d0:{terms}"),
    )
    expression, constants = symbolise_numbers(expression)

    from src.compiler.ir_identities import (
        narrow_float64_to_float32, two_product_flavor_scope,
    )
    # A float32 ladder has no fma to spell, so it lowers under the split
    # flavour -- the same pairing the GPU lanes use.
    flavour = two_product_flavor_scope("split" if element else "fma")

    compiled = compile_sympy_equations(
        [sympy.Eq(sympy.Symbol("y"), expression)], name="probe",
    )
    statements, _needs_math = materialize_function_body(
        compiled.function, tensor_vocabulary=True,
    )
    assigned, loaded = set(), set()
    for node in ast.walk(ast.Module(body=statements, type_ignores=[])):
        if isinstance(node, ast.Name):
            (assigned if isinstance(node.ctx, ast.Store)
             else loaded).add(node.id)
    parameters = tuple(sorted(loaded - assigned))

    body = [_IndexX().visit(each) for each in statements]
    produced = body.pop().value
    body.append(ast.Assign(
        targets=[ast.Subscript(
            value=ast.Name(id="y", ctx=ast.Load()),
            slice=ast.Name(id="i", ctx=ast.Load()), ctx=ast.Store(),
        )],
        value=produced,
    ))
    loop = ast.For(
        target=ast.Name(id="i", ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[ast.Name(id="n", ctx=ast.Load())], keywords=[],
        ),
        body=body, orelse=[],
    )
    ordered = (*parameters, "n", "y")
    def _slice():
        if element is None:
            return ast.Constant(value=width)
        return ast.Tuple(
            elts=[ast.Constant(value=width),
                  ast.Name(id=element, ctx=ast.Load())], ctx=ast.Load())

    annotate = (
        (lambda: ast.Subscript(
            value=ast.Name(id="Precision", ctx=ast.Load()),
            slice=_slice(), ctx=ast.Load(),
        ))
        if width > 1 else (lambda: None)
    )
    function = ast.FunctionDef(
        name="probe",
        args=ast.arguments(
            posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[],
            args=[
                ast.arg(arg=name,
                        annotation=None if name == "n" else annotate())
                for name in ordered
            ],
        ),
        body=[loop, ast.Return(value=ast.Name(id="y", ctx=ast.Load()))],
        decorator_list=[], returns=None, type_params=[],
    )
    source = ast.unparse(ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    ))

    with flavour:
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "probe", name=f"p{width}{element or ''}",
        )
    if element:
        narrow_float64_to_float32(module)
    entry = f"p{width}{element or ''}__probe"
    native = deploy(
        backend, module, entry, Path(f"build/probe_folded/{backend}_w{width}"),
    )
    ids = parameter_ids(module.functions[entry])
    return native, ids, sine, cosine, constants, terms


def score(width: int, digits: int, backend: str = "c", quiet: bool = True,
          element: str | None = None):
    native, ids, sine, cosine, constants, terms = build(
        width, digits, backend, element)

    quarter_exact = constant_rational("tau", digits) / 4
    turn = float(quarter_exact)
    points = np.array([
        0.0, 0.3, -0.3, 1.6, -1.6, 3.0, -3.0, 4.7, -4.7,
        17.25, -17.25, 100.0, -100.0, 1.0e6, -1.0e6, 1.0e12,
        turn, 2 * turn, 3 * turn, 5 * turn,
    ])
    count = len(points)

    def wide(value):
        return tuple(
            float(part) for part in
            limb_decomposition(value, width, element=element)
        )

    scalars = {
        "quarter": wide(quarter_exact),
        "neg_quarter": wide(-quarter_exact),
        "inv_quarter": wide(1 / quarter_exact),
    }
    for prefix, values in (("c", sine), ("d", cosine)):
        for index, value in enumerate(values):
            scalars[f"{prefix}{index}"] = wide(value)
    for name, value in constants.items():
        scalars[name] = wide(value)

    buffer = np.zeros(count * width)
    buffer[::width] = points
    feeds = {
        int(ids["n"]): np.int32(count),
        int(ids["x"]): buffer,
        int(ids["y"]): np.zeros(count * width),
    }
    for name, parts in scalars.items():
        if name not in ids:
            continue
        feeds[int(ids[name])] = np.float64(parts[0])
        for position in range(1, width):
            feeds[int(ids[f"{name}__limb{position}"])] = np.float64(
                parts[position]
            )

    execution = prepare(native, feeds)
    execution.run()
    produced = np.asarray(
        execution.buffers[int(ids["y"])]
    ).reshape(-1, width)

    mpmath.mp.dps = max(40, 20 * width)
    worst = 0.0
    for index, point in enumerate(points):
        got = sum(
            mpmath.mpf(float(produced[index][limb]))
            for limb in range(width)
        )
        truth = mpmath.sin(mpmath.mpf(float(point)))
        error = abs(got - truth)
        worst = max(worst, float(error))
        if not quiet:
            print(f"{point:>16} err={float(error):.3e}")
    return worst, terms


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "c"
    print(f"{'limbs':>6} {'digits':>7} {'terms':>6} {'worst error':>14}")
    for limbs in (1, 2, 3, 4):
        digits = max(17, 16 * limbs)
        try:
            worst, terms = score(limbs, digits, backend)
            print(f"{limbs:>6} {digits:>7} {terms:>6} {worst:14.3e}")
        except Exception as error:  # noqa: BLE001
            print(f"{limbs:>6} {digits:>7} {'-':>6} {str(error)[:40]:>14}")
