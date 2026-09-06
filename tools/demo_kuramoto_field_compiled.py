"""The same SymPy-derived Kuramoto step, compiled and run over whole arrays.

[tools/demo_kuramoto_field.py] proved the mathematics: build the equation
in SymPy, materialise it through ``compile_sympy_equations`` and
``materialize_function_body(tensor_vocabulary=True)`` into real
AbstractTensor Python, and let width alone decide precision. That version
runs it EAGERLY -- one Python-level operator dispatch per ``+``/``*``/
``.floor()`` -- which is why it measured ~570 ns per operator rather than
the ~10-30 ns a compiled sine core gets. Per standing instruction, the
speed of live code was never the point; the speed of COMPILED code is.

This file compiles the identical per-cell expression -- it is imported,
not re-derived -- into a WHOLE-FIELD kernel: the per-cell statements are
wrapped in one ``for i in range(n)`` loop over indexed array reads and a
single write to ``out[i]``, annotated ``Precision[width]`` exactly as the
proven sine cores are, and handed to ``lower_ast_source_to_ssa`` and a
native backend. One compiled call then advances every cell -- this is the
"batched" ask: the sines for the whole field happen inside one native
call, not one Python call per cell or per sine.

Only the torus WRAP is left to host code, and only because it is a gather
-- a decision about which memory holds a cell's neighbour, not arithmetic
on the numbers themselves. It moves every limb together, so it cannot
round and does not touch precision.

Run::

    python -m tools.demo_kuramoto_field_compiled
    python -m tools.demo_kuramoto_field_compiled --limbs 2 --backend c
"""

from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.demo_kuramoto_field import (  # noqa: E402
    NEIGHBOURS, core_terms, kuramoto_equation, local_coherence, render,
)
from src.common.tensors.signal_symbolic import (  # noqa: E402
    constant_rational, limb_decomposition,
)

#: The per-cell symbols that are FIELDS -- one value per cell -- as opposed
#: to scalars shared by every cell. Everything else ``kuramoto_equation``
#: names is a scalar: a coefficient, a derived constant, or a parameter.
FIELD_NAMES = ("theta", "omega", *NEIGHBOURS)


class _IndexFields(ast.NodeTransformer):
    """Rewrite a per-cell field symbol into an indexed array read.

    ``theta`` becomes ``theta[i]``. Nothing else changes -- this is the
    only difference between the scalar body one cell trusts and the
    array body every cell shares, and it is applied to the SAME
    statements the eager version runs, not a rewritten copy of them.
    """

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in FIELD_NAMES and isinstance(node.ctx, ast.Load):
            return ast.Subscript(
                value=ast.Name(id=node.id, ctx=ast.Load()),
                slice=ast.Name(id="i", ctx=ast.Load()),
                ctx=ast.Load(),
            )
        return node


def loop_source(equation, width: int,
                element: str | None = None) -> tuple[str, tuple[str, ...]]:
    """The whole-field kernel, as source, for one equation and one width.

    Same materialisation ``demo_kuramoto_field.materialise`` uses, stopped
    one step earlier -- before it is wrapped in a scalar ``FunctionDef``
    and exec'd -- so the statements can be placed inside a loop instead.

    ``element`` names the limb element when it is not the binary64 default.
    ``Precision[8, float32]`` is eight 24-bit limbs rather than eight
    53-bit ones, which is the ladder the GPU lanes carry because WGSL has
    no f64 at all -- there the choice is not "narrow or wide" but "one f32
    or several", and several is the one that keeps the mathematics.
    """

    from src.compiler.ssa_python_materializer import materialize_function_body
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations

    compiled = compile_sympy_equations([equation], name="kuramoto_step")
    statements, needs_math = materialize_function_body(
        compiled.function, tensor_vocabulary=True,
    )
    if needs_math:
        raise RuntimeError(
            "kuramoto_step: the materialised body wants the math module, "
            "which means a scalar opcode reached a tensor program"
        )

    assigned, loaded = set(), set()
    for node in ast.walk(ast.Module(body=statements, type_ignores=[])):
        if isinstance(node, ast.Name):
            (assigned if isinstance(node.ctx, ast.Store) else loaded).add(
                node.id
            )
    parameters = tuple(sorted(loaded - assigned))

    indexer = _IndexFields()
    body = [indexer.visit(statement) for statement in statements]
    assert isinstance(body[-1], ast.Return)
    result = body.pop().value
    body.append(ast.Assign(
        targets=[ast.Subscript(
            value=ast.Name(id="out", ctx=ast.Load()),
            slice=ast.Name(id="i", ctx=ast.Load()),
            ctx=ast.Store(),
        )],
        value=result,
    ))

    loop = ast.For(
        target=ast.Name(id="i", ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[ast.Name(id="n", ctx=ast.Load())], keywords=[],
        ),
        body=body, orelse=[],
    )

    def _subscript():
        if element is None:
            return ast.Constant(value=width)
        return ast.Tuple(
            elts=[ast.Constant(value=width), ast.Name(id=element,
                                                      ctx=ast.Load())],
            ctx=ast.Load(),
        )

    annotate = (
        (lambda: ast.Subscript(
            value=ast.Name(id="Precision", ctx=ast.Load()),
            slice=_subscript(), ctx=ast.Load(),
        ))
        if width > 1 else (lambda: None)
    )
    ordered = (*parameters, "n", "out")
    function = ast.FunctionDef(
        name="kuramoto_step",
        args=ast.arguments(
            posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[],
            args=[
                ast.arg(arg=name, annotation=None if name == "n" else annotate())
                for name in ordered
            ],
        ),
        body=[loop, ast.Return(value=ast.Name(id="out", ctx=ast.Load()))],
        decorator_list=[], returns=None, type_params=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    return ast.unparse(module), ordered


def deploy(backend: str, module, entry: str, directory: Path):
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, entry)
        if not artifact.complete:
            raise RuntimeError("; ".join(
                f"{item.operation}: {item.reason}"
                for item in artifact.shortfalls[:3]
            ))
        return artifact.compile(directory)
    if backend == "llvm":
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
        )

        artifact = emit_ssa_function_to_llvm(module, entry)
        if artifact.shortfalls:
            raise RuntimeError("; ".join(
                item.reason[:90] for item in artifact.shortfalls[:3]
            ))
        return compile_artifact(artifact, directory=directory)
    if backend == "fortran":
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        from src.compiler.ssa_fortran_backend import (
            FortranCoreNative, emit_module,
        )

        fortran = emit_module(module, progress=lambda _line: None)
        if not fortran.complete:
            raise RuntimeError("; ".join(
                f"{item.operation}: {item.reason}"
                for item in fortran.shortfalls[:3]
            ))
        built = compile_fortran_module_c_shell(
            fortran, {}, directory, library=True, entrypoint=entry,
            name="kuramoto",
        )
        record = next(
            each for each in fortran.api.entry_points
            if str(each.name) == entry
        )
        return FortranCoreNative(built.executable_path, record)
    raise RuntimeError(f"unknown backend {backend!r}")


def prepare(native, feeds):
    preparer = getattr(native, "prepare_execution", None)
    if preparer is not None:
        return preparer(feeds)
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    return prepare_artifact_execution(native, feeds)


def parameter_ids(function) -> dict:
    ids = dict(function.metadata["parameter_names"])
    rows = dict(function.metadata.get("precision_lowered_values") or ())
    for name, identifier in tuple(ids.items()):
        row = rows.get(int(identifier))
        if row:
            for position, limb in enumerate(tuple(row)[1:], start=1):
                ids.setdefault(f"{name}__limb{position}", int(limb))
    return ids


def interleaved(values: np.ndarray, width: int) -> np.ndarray:
    """One double per limb per element, the compiler's own array ABI."""

    buffer = np.zeros(values.size * width)
    buffer[::width] = values.ravel()
    return buffer


def rolled(buffer: np.ndarray, width: int, height: int, limbs: int,
          axis: int, shift: int) -> np.ndarray:
    """A neighbour field: the interleaved buffer, gathered by a torus roll.

    A roll is a copy -- it cannot round -- so moving every limb together
    keeps the expansion intact. This is the whole of the torus, expressed
    as data movement rather than arithmetic.
    """

    grid = buffer.reshape(height, width, limbs)
    return np.roll(grid, shift, axis=axis).reshape(-1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--coupling", type=float, default=0.8)
    parser.add_argument("--spread", type=float, default=0.4)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--limbs", type=int, default=1)
    parser.add_argument("--digits", type=int, default=None)
    parser.add_argument("--backend", default="llvm",
                        choices=("llvm", "c", "fortran"))
    parser.add_argument("--output", type=Path,
                        default=Path("build/kuramoto-field-compiled"))
    arguments = parser.parse_args(argv)

    width, height = int(arguments.width), int(arguments.height)
    cells = width * height
    limbs = max(1, int(arguments.limbs))
    digits = int(arguments.digits or max(17, 16 * limbs))

    print(
        f"field {width}x{height} = {cells:,d} cells, {arguments.steps} "
        f"steps, {limbs} limb(s), {digits} digits, {arguments.backend} lane",
        flush=True,
    )

    sine = list(core_terms("sin", digits))
    cosine = list(core_terms("cos", digits))
    terms = max(len(sine), len(cosine))
    sine += [Fraction(0)] * (terms - len(sine))
    cosine += [Fraction(0)] * (terms - len(cosine))

    equation, constants = kuramoto_equation(terms)
    source, ordered = loop_source(equation, limbs)
    print(
        f"SymPy -> whole-field kernel: {len(source.splitlines())} lines, "
        f"{terms} terms per series, {len(ordered)} parameters",
        flush=True,
    )

    started = time.perf_counter()
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "kuramoto_step", name="kur",
    )
    entry = "kur__kuramoto_step"
    native = deploy(
        arguments.backend, module, entry,
        arguments.output / f"{arguments.backend}_w{limbs}",
    )
    print(f"compiled in {time.perf_counter() - started:.1f}s", flush=True)

    ids = parameter_ids(module.functions[entry])

    generator = np.random.default_rng(int(arguments.seed))
    phases = generator.uniform(-math.pi, math.pi, cells)
    spin = generator.normal(0.0, float(arguments.spread), cells)

    quarter_exact = constant_rational("tau", digits) / 4

    def wide(value) -> tuple:
        return tuple(float(part) for part in limb_decomposition(value, limbs))

    scalars = {
        "coupling": wide(Fraction(arguments.coupling)),
        "dt": wide(Fraction(arguments.dt)),
        "quarter": wide(quarter_exact),
        "neg_quarter": wide(-quarter_exact),
        "inv_quarter": wide(1 / quarter_exact),
    }
    for prefix, values in (("c", sine), ("d", cosine)):
        for index, value in enumerate(values):
            scalars[f"{prefix}{index}"] = wide(value)
    for name, value in constants.items():
        scalars[name] = wide(value)

    theta_buffer = interleaved(phases, limbs)
    omega_buffer = interleaved(spin, limbs)
    out_buffer = np.zeros(cells * limbs)

    feeds = {int(ids["n"]): np.int32(cells)}
    for name, parts in scalars.items():
        if name not in ids:
            continue
        feeds[int(ids[name])] = np.float64(parts[0])
        for position in range(1, limbs):
            feeds[int(ids[f"{name}__limb{position}"])] = np.float64(
                parts[position]
            )
    feeds[int(ids["theta"])] = theta_buffer
    feeds[int(ids["omega"])] = omega_buffer
    feeds[int(ids["out"])] = out_buffer
    directions = {"up": (0, 1), "down": (0, -1), "left": (1, 1),
                  "right": (1, -1)}
    for name in NEIGHBOURS:
        feeds[int(ids[name])] = np.zeros(cells * limbs)

    execution = prepare(native, feeds)
    theta_view = np.asarray(execution.buffers[int(ids["theta"])])
    out_view = np.asarray(execution.buffers[int(ids["out"])])
    neighbour_views = {
        name: np.asarray(execution.buffers[int(ids[name])])
        for name in NEIGHBOURS
    }

    print()
    print(f"{'step':>6}  {'coherence':>10}  {'seconds':>9}", flush=True)
    frames = []
    interval = max(1, int(arguments.steps) // 5)
    compiled_seconds = 0.0
    for index in range(int(arguments.steps) + 1):
        current = theta_view.reshape(-1, limbs)[:, 0].copy()
        if index % interval == 0:
            print(
                f"{index:6d}  {local_coherence(current, width, height):10.4f}"
                f"  {compiled_seconds:9.2f}",
                flush=True,
            )
            frames.append(render(current, width, height))
        if index == int(arguments.steps):
            break

        for name, (axis, shift) in directions.items():
            neighbour_views[name][...] = rolled(
                theta_view, width, height, limbs, axis, shift,
            )
        moment = time.perf_counter()
        execution.run()
        compiled_seconds += time.perf_counter() - moment
        theta_view[...] = out_view

    sines = 4 * cells * int(arguments.steps)
    print()
    print(
        f"{sines:,d} sines in {compiled_seconds:.2f}s compiled "
        f"({compiled_seconds * 1e9 / max(sines, 1):.1f} ns each, "
        f"{cells / (compiled_seconds / max(int(arguments.steps), 1)) / 1e6:.2f} "
        f"million cells/second, one native call per step)",
        flush=True,
    )

    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "kuramoto_step_loop.py").write_text(source, encoding="utf-8")
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
            f"Kuramoto field, compiled {arguments.backend} lane, "
            f"{limbs} limb(s), one native call per step"
        )
        figure.tight_layout()
        path = destination / "kuramoto_field_compiled.png"
        figure.savefig(path, dpi=110)
        pyplot.close(figure)
        print(f"wrote {path}", flush=True)
    except ImportError:
        print("matplotlib is absent; skipped the picture", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
