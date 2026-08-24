"""A field of oscillators, evolved by a program SymPy wrote.

The update is stated once, as MATHEMATICS::

    dtheta/dt = omega + K * sum over neighbours sin(their phase - mine)

and nothing here implements it. The expression is built in SymPy --
including the range reduction and the sine and cosine series, whose
coefficients stay SYMBOLIC -- handed to ``compile_sympy_equations``, and
materialised by ``materialize_function_body(tensor_vocabulary=True)`` into
real AbstractTensor Python. That materialised program is what runs. It is
printed at startup so there is nothing to take on trust.

Because the coefficients never become literals, the emitted body holds no
number at all: it is pure shape. WHAT THE CALLER PASSES DECIDES THE
PRECISION. Hand it ``Precision`` operands and every operator in it becomes
limbed arithmetic through the same tree of calls, so one materialisation
serves the whole ladder and ``--limbs`` is the only thing that changes.

Two details are worth naming, because both are what make the field an
honest exercise of the pack rather than a flattering one:

* The coupling needs ``sin`` of phase DIFFERENCES, which land anywhere on
  the real line, while the series is proven only near zero. So the
  expression folds its own argument onto the nearest quarter turn first.
* Which quarter was folded away is selected by an exact LAGRANGE basis on
  {0,1,2,3} -- four cubics in the quadrant index, exact at those four
  points. No comparison, no branch, no mask: the selection is arithmetic,
  so it is limbed like everything else and needs nothing from the backend
  that the rest of the expression does not already need.

The field itself lives on a torus with a one-cell HALO, and the neighbour
fetch is an index gather per limb -- data movement, which cannot round.

Run::

    python -m tools.demo_kuramoto_field
    python -m tools.demo_kuramoto_field --limbs 2 --digits 32
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
import sys
import time

import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.tensors.abstraction import AbstractTensor  # noqa: E402
from src.common.tensors.extended_precision import Precision  # noqa: E402
from src.common.tensors.signal_symbolic import (  # noqa: E402
    CORE_RADII, constant_rational, limb_decomposition, order_for,
    order_to_degree, structured_coefficients,
)

NEIGHBOURS = ("up", "down", "left", "right")


def folded_sine(difference, quarter, neg_quarter, inv_quarter, sine, cosine):
    """``sin(difference)`` for any argument, as one SymPy expression.

    Fold onto the nearest quarter turn, evaluate both series on what is
    left, and select by the quarter that was removed.

    Two prior selections were tried and MEASURED wrong. An exact Lagrange
    cubic in the quadrant index was correct eagerly and at width one
    compiled, but at width two compiled it came back off by up to 122
    orders of magnitude -- the compiler's precision lowering does not
    carry a value cubed through a wide chain the way it carries a sum or
    product. ``Eq``/``Piecewise`` compiled to a branch that split the
    function into regions and then could not find its own operands across
    them -- the same SSA wall an early hand-authored attempt hit.

    What is used here selects by two PARITY BITS instead of a value in
    {0,1,2,3}: which series (``index`` even or odd) and which sign
    (``index // 2`` even or odd), each an exact one-limb 0/1 taken by the
    same floor/subtract this file already proves safe for the fold
    itself. Linear in ``index``, not cubic, and no comparison at all --
    the two failures above shared no cause, so the fix that survives both
    is avoiding the ingredient each one needed: a value raised past
    degree one, or a branch.

    A THIRD defect, found after both of those: ``difference - index *
    quarter`` looks harmless, but SymPy canonicalises subtraction to
    ``Add(difference, Mul(-1, index, quarter))``, and the materialiser
    groups that ``-1`` with ``quarter`` -- a WIDE FORMAL PARAMETER --
    before multiplying by ``index``. MEASURED: the compiled result of
    ``(-1) * quarter`` silently truncates to a plain integer downstream
    (confirmed by feeding ``quarter``'s low limb a 999.0 marker and
    watching the truncated product move by round numbers), so every
    residual came out as an integer count of quarter-turns instead of a
    small remainder. Multiplying a wide formal by a bare ``-1`` is the
    trigger; multiplying it by anything else measured fine. So the fix is
    to never form that product: ``neg_quarter`` arrives already negative,
    decomposed to limbs the same way every other constant here is, and
    the fold uses addition only.

    A FOURTH defect turned out to be the THIRD one again, wearing a
    different coat. At width two every point landing in the sine
    quadrants was exact while every point in the cosine quadrants came
    back as the sine branch -- the blend was selecting wrongly, and the
    residual it blended was measurably correct. The blend was written
    ``odd + parity * (even - odd)``, and ``even - odd`` is
    ``Add(even, Mul(-1, odd))``: a WIDE value negated, and then that
    negation multiplied. Exactly the shape that already had to be fixed
    for ``quarter``, only here the wide operand is a runtime value rather
    than a formal, so the earlier fix could not cover it.

    Written as a convex blend instead -- ``odd * other + even * parity``
    with ``other = 1 - parity`` -- both selectors are narrow, both
    products are narrow times wide, and no wide value is negated inside a
    product anywhere. The two selectors are exact complementary bits, so
    this is the same selection, spelled so the defect has nothing to bite
    on.
    """

    index = sympy.floor(difference * inv_quarter + sympy.Rational(1, 2))
    residual = difference + index * neg_quarter
    square = residual ** 2

    def horner(symbols):
        value = symbols[-1]
        for symbol in reversed(symbols[:-1]):
            value = symbol + square * value
        return value

    odd = residual * horner(sine)
    even = horner(cosine)

    base = difference * inv_quarter + sympy.Rational(1, 2)
    half = sympy.floor(base * sympy.Rational(1, 2))
    quarter_step = sympy.floor(base * sympy.Rational(1, 4))
    series_parity = index - 2 * half
    other_parity = 1 - series_parity
    sign_parity = half - 2 * quarter_step
    sign = 1 - 2 * sign_parity
    return sign * (odd * other_parity + even * series_parity)


def symbolise_numbers(expression, prefix: str = "k"):
    """Lift EVERY numeric constant out of the expression as a parameter.

    A number that survives into the materialised program is emitted as a
    Python literal -- one double if it is fractional, and an INTEGER if it
    is whole -- and no width downstream can repair either. Both failure
    modes were measured here, and the second one three separate times
    before its shape was recognised.

    The fractional case is the obvious one: a third rounds once, forever,
    so the error at two, three and four limbs came back identical to the
    last digit.

    The integer case is worse because it does not look like a number
    problem at all. SymPy canonicalises ``a - b`` to ``Add(a, Mul(-1, b))``
    and types that ``-1`` as an integer, which then TYPES THE PRODUCT --
    so a wide value multiplied by it inferred an integer result, and each
    backend went wrong in its own dialect. The C lane truncated a
    precision residual to a whole count of quarter turns; the same shape
    later selected the sine series where the cosine one was owed, in every
    quadrant that needed it; the Fortran lane emitted ``int(x, c_int)``
    and additionally overflowed 2**31, wrong by 5.1e+11 at an argument of
    1e12 while every smaller argument stayed exact. Three symptoms, one
    integer literal.

    So nothing numeric is left in the body. Every constant leaves as a
    symbol and comes back decomposed to the width in use, which is
    deliberate overkill for something like ``2`` and exactly the point:
    the program cannot be typed down by a constant it does not contain.
    The one number that must STAY is a ``Pow`` exponent, which is
    structure rather than a quantity -- ``r ** 2`` says how many factors,
    and a symbol there is not a different precision, it is a different
    program.

    Returns the rewritten expression and the exact value owed to each new
    parameter.
    """

    values: dict[str, Fraction] = {}
    cache: dict = {}

    def convert(node):
        if node.is_Number:
            held = cache.get(node)
            if held is not None:
                return held
            symbol = sympy.Symbol(f"{prefix}{len(values)}")
            values[symbol.name] = Fraction(int(node.p), int(node.q))
            cache[node] = symbol
            return symbol
        if node.is_Atom:
            return node
        if isinstance(node, sympy.Pow):
            base, exponent = node.args
            return sympy.Pow(convert(base), exponent)
        return node.func(*(convert(each) for each in node.args))

    return convert(expression), values


#: Which cell each neighbour symbol reads, as (row, column) steps on the
#: torus. The gather is generated from this, so a program that wants a
#: wider neighbourhood states it here rather than editing a shader.
KURAMOTO_STENCIL = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}


@dataclass(frozen=True)
class FieldProgram:
    """One lattice program: its mathematics, and what it needs from memory.

    Everything downstream -- materialisation, lowering, narrowing, WGSL
    emission, feed packing, the host -- consumes only this. Swapping the
    physics means writing another builder that returns one of these; it
    does not mean touching the compiler or the page.
    """

    name: str
    equation: object
    constants: dict
    #: Per-cell values held in memory between steps.
    state: tuple
    #: The state field the equation advances.
    advances: str
    #: Neighbour symbol -> (row, column) offset.
    stencil: dict
    #: Runtime scalars beyond the series and the derived constants.
    scalars: tuple


def kuramoto_program(terms: int, lag: bool = False) -> FieldProgram:
    """Kuramoto on a torus, optionally with a Sakaguchi phase lag.

    The lag is ONE TERM -- ``sin(difference - alpha)`` instead of
    ``sin(difference)`` -- and it changes the character of the field
    rather than its scale: at zero lag the neighbours pull straight
    toward agreement and the lattice locks into domains, while a nonzero
    lag makes the pull asymmetric, so a defect no longer sits still and
    the field carries spiral waves and can hold locked and drifting
    regions at once.

    That it is one term here, and nothing anywhere else, is the point of
    the FieldProgram seam.
    """

    theta, omega = sympy.symbols("theta omega")
    coupling, dt = sympy.symbols("coupling dt")
    quarter, neg_quarter, inv_quarter = sympy.symbols(
        "quarter neg_quarter inv_quarter"
    )
    alpha = sympy.Symbol("alpha")
    sine = sympy.symbols(f"c0:{terms}")
    cosine = sympy.symbols(f"d0:{terms}")

    def difference(name: str):
        gap = sympy.Symbol(name) - theta
        return gap - alpha if lag else gap

    pull = sum(
        folded_sine(
            difference(name), quarter, neg_quarter, inv_quarter,
            sine, cosine,
        )
        for name in KURAMOTO_STENCIL
    )
    advanced = theta + dt * (omega + coupling * pull)
    advanced, constants = symbolise_numbers(advanced)
    return FieldProgram(
        name="sakaguchi" if lag else "kuramoto",
        equation=sympy.Eq(sympy.Symbol("advanced"), advanced),
        constants=constants,
        state=("theta", "omega"),
        advances="theta",
        stencil=dict(KURAMOTO_STENCIL),
        scalars=("coupling", "dt") + (("alpha",) if lag else ()),
    )


def kuramoto_equation(terms: int):
    """The advance alone, for callers that only want the mathematics."""

    program = kuramoto_program(terms)
    return program.equation, program.constants


def materialise(equation, name: str):
    """SymPy in, AbstractTensor Python out. The same route the cores take.

    No source is templated: the equation goes to the symbolic compiler,
    the SSA it produces goes to the materializer under the tensor
    vocabulary, and what comes back is a real module compiled from a real
    AST. Returns ``(callable, parameter names, source)``.
    """

    from src.compiler.ssa_python_materializer import materialize_function_body
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations

    compiled = compile_sympy_equations([equation], name=name)
    statements, needs_math = materialize_function_body(
        compiled.function, tensor_vocabulary=True,
    )
    if needs_math:
        raise RuntimeError(
            f"{name}: the materialised body wants the math module, which "
            f"means a scalar opcode reached a tensor program"
        )

    assigned, loaded = set(), set()
    for node in ast.walk(ast.Module(body=statements, type_ignores=[])):
        if isinstance(node, ast.Name):
            (assigned if isinstance(node.ctx, ast.Store) else loaded).add(
                node.id
            )
    parameters = tuple(sorted(loaded - assigned))

    function = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg=each) for each in parameters],
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=statements, decorator_list=[], returns=None, type_params=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    namespace: dict = {}
    exec(compile(module, f"<{name}>", "exec"), namespace)
    return namespace[name], parameters, ast.unparse(module)


def core_terms(name: str, digits: int):
    """The series this tree derived, as exact rationals."""

    order = order_to_degree(
        name, order_for(name, CORE_RADII[name], digits=digits)
    )
    return structured_coefficients(name, order)


def shifted(field: Precision, gather) -> Precision:
    """A neighbour field: the same values, read from somewhere else.

    A gather moves data and cannot round, so it distributes over the
    expansion -- which is exactly the contract ``_map_limbs`` states.
    """

    return field._map_limbs(lambda limb: limb.reshape(-1)[gather])


def torus_gathers(width: int, height: int):
    """Index arrays that read each of the four neighbours, wrapped."""

    grid = np.arange(width * height).reshape(height, width)
    return {
        "up": np.roll(grid, 1, axis=0).ravel(),
        "down": np.roll(grid, -1, axis=0).ravel(),
        "left": np.roll(grid, 1, axis=1).ravel(),
        "right": np.roll(grid, -1, axis=1).ravel(),
    }


def local_coherence(theta, width: int, height: int) -> float:
    """How well each cell agrees with the neighbours it can feel.

    The classic order parameter averages over the whole field and reads
    near zero however well it has organised, because locally coupled
    oscillators lock into DOMAINS whose phases cancel. This asks the
    question the coupling actually answers. It is a MEASUREMENT of the
    result, so libm serving it costs the field nothing.
    """

    grid = theta.reshape(height, width)
    return float(np.mean([
        np.mean(np.cos(np.roll(grid, shift, axis=axis) - grid))
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1))
    ]))


def render(theta, width: int, height: int):
    """Phase as hue, because phase wraps and a linear ramp would seam."""

    grid = np.mod(theta.reshape(height, width), 2.0 * math.pi) / (
        2.0 * math.pi
    )
    return np.clip(np.stack([
        0.5 + 0.5 * np.cos(2.0 * math.pi * (grid - shift))
        for shift in (0.0, 1.0 / 3.0, 2.0 / 3.0)
    ], axis=-1), 0.0, 1.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--coupling", type=float, default=0.8)
    parser.add_argument("--spread", type=float, default=0.4)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--limbs", type=int, default=1,
                        help="width of the arithmetic; the only knob the "
                             "materialised program responds to")
    parser.add_argument("--digits", type=int, default=None,
                        help="core size; defaults to what the width holds")
    parser.add_argument("--show-source", action="store_true", default=True)
    parser.add_argument("--output", type=Path,
                        default=Path("build/kuramoto-field"))
    arguments = parser.parse_args(argv)

    width, height = int(arguments.width), int(arguments.height)
    cells = width * height
    limbs = max(1, int(arguments.limbs))
    # The coefficients of a tier are the depth of the tier: a width that
    # can hold more digits than the series carries buys nothing, and a
    # series longer than the width can hold is silently truncated.
    digits = int(arguments.digits or max(17, 16 * limbs))

    print(
        f"field {width}x{height} = {cells:,d} cells, {arguments.steps} "
        f"steps, {limbs} limb(s), {digits} digits",
        flush=True,
    )

    sine = list(core_terms("sin", digits))
    cosine = list(core_terms("cos", digits))
    terms = max(len(sine), len(cosine))
    # One Horner loop serves both series, so the shorter is extended with
    # the zeros its structure already implies.
    sine += [Fraction(0)] * (terms - len(sine))
    cosine += [Fraction(0)] * (terms - len(cosine))

    started = time.perf_counter()
    equation, constants = kuramoto_equation(terms)
    step, parameters, source = materialise(equation, "kuramoto_step")
    print(
        f"SymPy -> AbstractTensor Python in "
        f"{time.perf_counter() - started:.1f}s: {len(source.splitlines())} "
        f"lines, {terms} terms per series, {len(parameters)} parameters",
        flush=True,
    )
    if arguments.show_source:
        print()
        print(source[:1400] + ("\n..." if len(source) > 1400 else ""))
        print()

    generator = np.random.default_rng(int(arguments.seed))
    phases = generator.uniform(-math.pi, math.pi, cells)
    spin = generator.normal(0.0, float(arguments.spread), cells)

    theta = Precision.of(AbstractTensor.get_tensor(phases), limbs)
    omega = Precision.of(AbstractTensor.get_tensor(spin), limbs)

    quarter_exact = constant_rational("tau", digits) / 4
    supply = {
        "omega": omega,
        "coupling": Precision.constant(theta, tuple(
            float(part) for part in
            limb_decomposition(Fraction(arguments.coupling), limbs)
        )),
        "dt": Precision.constant(theta, tuple(
            float(part) for part in
            limb_decomposition(Fraction(arguments.dt), limbs)
        )),
        "quarter": Precision.constant(theta, tuple(
            float(part) for part in limb_decomposition(quarter_exact, limbs)
        )),
        "neg_quarter": Precision.constant(theta, tuple(
            float(part) for part in
            limb_decomposition(-quarter_exact, limbs)
        )),
        # The reciprocal only has to land the fold in the right quarter;
        # its error is removed by the exact subtraction that follows, so
        # one limb of it is enough and more would be wasted work.
        "inv_quarter": Precision.constant(theta, tuple(
            float(part) for part in
            limb_decomposition(1 / quarter_exact, limbs)
        )),
    }
    for prefix, values in (("c", sine), ("d", cosine)):
        for index, value in enumerate(values):
            supply[f"{prefix}{index}"] = Precision.constant(theta, tuple(
                float(part) for part in limb_decomposition(value, limbs)
            ))
    # The fractions the expression itself needed, at the same depth as
    # everything else -- this is the difference between width buying
    # precision and width buying nothing.
    for name, value in constants.items():
        supply[name] = Precision.constant(theta, tuple(
            float(part) for part in limb_decomposition(value, limbs)
        ))

    gathers = torus_gathers(width, height)
    missing = set(parameters) - set(supply) - {"theta", *NEIGHBOURS}
    if missing:
        raise RuntimeError(f"unsupplied parameters: {sorted(missing)}")

    print(f"{'step':>6}  {'coherence':>10}  {'spread':>10}  {'seconds':>9}",
          flush=True)
    frames = []
    interval = max(1, int(arguments.steps) // 5)
    elapsed = 0.0
    for index in range(int(arguments.steps) + 1):
        current = np.asarray(theta.collapse().tolist(), dtype=float).ravel()
        if index % interval == 0:
            print(
                f"{index:6d}  {local_coherence(current, width, height):10.4f}"
                f"  {float(np.std(current)):10.4f}  {elapsed:9.2f}",
                flush=True,
            )
            frames.append(render(current, width, height))
        if index == int(arguments.steps):
            break

        arguments_for_step = {
            "theta": theta,
            **{name: shifted(theta, gathers[name]) for name in NEIGHBOURS},
            **supply,
        }
        moment = time.perf_counter()
        theta = step(**{
            name: arguments_for_step[name] for name in parameters
        })
        elapsed += time.perf_counter() - moment

    print()
    print(
        f"{4 * cells * int(arguments.steps):,d} sines at {limbs} limb(s), "
        f"every one folding its own argument onto the quarter turn",
        flush=True,
    )

    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "kuramoto_step.py").write_text(source, encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        figure, axes = pyplot.subplots(
            1, len(frames), figsize=(2.7 * len(frames), 3.1),
        )
        for position, (panel, image) in enumerate(zip(axes, frames)):
            panel.imshow(image, interpolation="nearest")
            panel.set_title(f"step {position * interval}")
            panel.axis("off")
        figure.suptitle(
            f"Kuramoto field, phase as hue -- SymPy through AbstractTensor "
            f"at {limbs} limb(s)"
        )
        figure.tight_layout()
        path = destination / "kuramoto_field.png"
        figure.savefig(path, dpi=110)
        pyplot.close(figure)
        print(f"wrote {path} and {destination / 'kuramoto_step.py'}",
              flush=True)
    except ImportError:
        print("matplotlib is absent; skipped the picture", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
