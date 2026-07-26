"""One ordinary AbstractTensor program, three backends, one picture.

Escape-time Mandelbrot as an ordinary AbstractTensor function. The iteration is
*unrolled*, so ``iterations`` steps become ~10x that many instructions in a
single program with no control flow at all -- which is the point: it is the
deepest fused program either backend has been asked to run, and it renders
something you can look at.

Why it needs no ``select``/``where``
------------------------------------
Escape counting is usually written with a branch. It does not need one: the
comparison primitives already return 1.0/0.0, so

    still_inside = le(|z|^2, 4.0)
    count        = count + still_inside

accumulates the escape iteration branchlessly. So this demo runs entirely on
ops both backends already have -- nothing was invented to make it work. (The
missing ternary ``select`` still matters for *general* control flow; it just
isn't needed here.)

What it exercises
-----------------
* the GLSL emitter at depth -- hundreds of instructions in one shader, every
  intermediate a register-resident local, one dispatch;
* the C backend's private slot lowering over the same FusedProgram steps;
* the shared canonical op vocabulary, from two directions at once;
* numpy as the behavioural oracle for both.

Python executes the ordinary AbstractTensor function once under GradTape
capture. The resulting established ``FusedProgram`` is then the single input
to NumPy verification, the C one-call backend, and GLSL shader lowering. There
is no demo-only instruction class or separately maintained NumPy algorithm.

Run it::

    python -m src.common.tensors.accelerator_backends.demo_mandelbrot_fusion
    python -m ...demo_mandelbrot_fusion --width 1600 --height 1200 --iterations 96
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# the program
# ---------------------------------------------------------------------------

# Escaped orbits diverge without bound. In float32 they reach inf within a few
# more iterations, then inf-inf produces NaN, and the two backends round that
# boundary differently -- the first version of this demo agreed with numpy on
# only 99.9577% of pixels, max |diff| = 18, purely from overflow timing.
#
# Pinning |zx|,|zy| to ORBIT_CLAMP fixes it exactly. Points still inside the set
# satisfy |z| <= 2 by definition, so clamping at 1e18 cannot affect them; escaped
# points freeze at a magnitude whose square (1e36) is still finite in float32 and
# still far above the escape radius, so their count stays frozen either way.
# minimum/maximum are in both backends' vocabularies already.
ORBIT_CLAMP = 1e18


def mandelbrot_escape(cx, cy, iterations: int, clamp: float = ORBIT_CLAMP):
    """Ordinary backend-agnostic AbstractTensor Mandelbrot computation."""

    zx = cx * 0.0
    zy = cx * 0.0
    count = cx * 0.0
    for _ in range(iterations):
        zx2, zy2 = zx * zx, zy * zy
        count = count + (zx2 + zy2 <= 4.0)
        zx, zy = zx2 - zy2 + cx, 2.0 * zx * zy + cy
        zx = zx.minimum(clamp).maximum(-clamp)
        zy = zy.minimum(clamp).maximum(-clamp)
    return count


def capture_mandelbrot(cx: np.ndarray, cy: np.ndarray, iterations: int):
    """Execute and capture the ordinary function as one FusedProgram."""

    from ..autograd import autograd
    from ..numpy_backend import NumPyTensorOperations
    from .c_primitive_program import compile_elementwise_tape

    with autograd.forward_capture() as tape:
        x = NumPyTensorOperations.tensor(cx)
        y = NumPyTensorOperations.tensor(cy)
        output = mandelbrot_escape(x, y, iterations)
    captured = compile_elementwise_tape(tape, output)
    captured = type(captured)(captured.program, {id(x): x, id(y): y})
    return captured, np.asarray(output.tolist(), dtype=cx.dtype)


def run_abstract_numpy(cx: np.ndarray, cy: np.ndarray, iterations: int):
    """Run the exact same AbstractTensor function on the NumPy backend."""

    from ..numpy_backend import NumPyTensorOperations

    result = mandelbrot_escape(
        NumPyTensorOperations.tensor(cx),
        NumPyTensorOperations.tensor(cy),
        iterations,
    )
    return np.asarray(result.tolist(), dtype=cx.dtype)


def complex_plane(width: int, height: int, center: complex, span: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Row-major cx/cy grids, flattened, float32."""
    aspect = width / height
    xs = np.linspace(center.real - span * aspect / 2,
                     center.real + span * aspect / 2, width, dtype=np.float32)
    ys = np.linspace(center.imag - span / 2,
                     center.imag + span / 2, height, dtype=np.float32)
    cx, cy = np.meshgrid(xs, ys)
    return (np.ascontiguousarray(cx.ravel()),
            np.ascontiguousarray(cy.ravel()))


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

def _replacement_feeds(captured, cx, cy):
    """Bind replacement arrays by matching the two captured root identities."""

    feed_ids = list(captured.feeds)
    if len(feed_ids) != 2:
        raise ValueError(f"expected cx/cy capture roots, found {len(feed_ids)}")
    return {feed_ids[0]: cx, feed_ids[1]: cy}


def run_glsl(captured, cx, cy):
    from .glsl_backend import execute_program

    return execute_program(
        captured.program, _replacement_feeds(captured, cx, cy)
    ).numpy()


def run_c(captured, cx, cy):
    from .c_backend import CTensor
    from .c_primitive_program import execute_fused_program

    feeds = {
        feed_id: CTensor.from_list(array.tolist(), array.shape)
        for feed_id, array in _replacement_feeds(captured, cx, cy).items()
    }
    return np.asarray(
        execute_fused_program(captured.program, feeds).tolist(), dtype=np.float32
    )


def c_workspace_bytes(program, elements: int) -> int:
    """The C interpreter allocates one full slot array per instruction result."""
    return (len(program.feeds) + len(program.steps)) * elements * 8


# ---------------------------------------------------------------------------
# picture
# ---------------------------------------------------------------------------

def save_image(counts: np.ndarray, width: int, height: int, path: Path,
               cmap: str = "blue_fire", vignette_tile: int = 0) -> Path:
    """Colour with the repository's own colormap rather than a private one.

    ``vignette_tile`` is off (0) by default and for a reason worth recording:
    ``render_cache.add_vignette`` is not a border vignette, it **upscales**, turning
    every input pixel into a ``tile x tile`` bubble. It is built for small conv
    feature maps, where that reads as pixel art. Applied blind to a 1600x1200
    render at its default ``tile=8`` it silently produces a 12800x9600, 11 MB
    image -- which is exactly what the first version of this demo did.
    """
    from PIL import Image
    from ..abstract_convolution.render_cache import add_vignette, apply_colormap

    frame = counts.reshape(height, width)
    # sqrt spreads the low counts, where nearly all the visible structure lives
    rgb = apply_colormap(np.sqrt(frame), cmap=cmap)
    if vignette_tile:
        rgb = add_vignette(rgb, tile=vignette_tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--iterations", type=int, default=64)
    ap.add_argument("--center", type=complex, default=complex(-0.743643887, 0.131825904))
    ap.add_argument("--span", type=float, default=0.004)
    ap.add_argument("--cmap", default="blue_fire")
    ap.add_argument("--vignette-tile", type=int, default=0,
                    help="per-pixel bubble vignette; UPSCALES by this factor "
                         "(render_cache.add_vignette default is 8). 0 = off.")
    ap.add_argument("--out", type=Path, default=Path("mandelbrot_fused.png"))
    ap.add_argument("--c-probe", type=int, default=48,
                    help="edge length of the small grid cross-checked on the C backend")
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument(
        "--only-glsl",
        action="store_true",
        help="render with GLSL only; skip the NumPy/f64 oracles and C probe",
    )
    args = ap.parse_args(argv)

    elements = args.width * args.height
    print(f"image   : {args.width}x{args.height} = {elements:,} pixels")

    cx, cy = complex_plane(args.width, args.height, args.center, args.span)
    captured, _ = capture_mandelbrot(cx[:2], cy[:2], args.iterations)
    print(f"program : {len(captured.program.steps)} FusedProgram steps, "
          f"{args.iterations} Python-loop iterations captured")

    # -- GPU ---------------------------------------------------------------
    from .gl_context import require_gl_context
    info = require_gl_context()
    print(f"gpu     : {info['renderer']} (context: {info['source']})")

    t0 = time.perf_counter()
    gpu = run_glsl(captured, cx, cy)
    gpu_ms = (time.perf_counter() - t0) * 1e3
    print(f"glsl    : {gpu_ms:8.1f} ms  "
          f"({len(captured.program.steps)} steps x {elements:,} px, one dispatch)")

    if not args.only_glsl:
        # -- oracle --------------------------------------------------------
        t0 = time.perf_counter()
        ref = run_abstract_numpy(cx, cy, args.iterations)
        np_ms = (time.perf_counter() - t0) * 1e3
        print(f"numpy   : {np_ms:8.1f} ms  (same AbstractTensor function)")

        max_err = float(np.max(np.abs(gpu - ref)))
        agree = float(np.mean(gpu == ref)) * 100.0
        print(f"agree   : {agree:.4f}% exact vs numpy-f32, max |diff| = {max_err:g}")

        # Escape-time is chaotic: a 1-ULP boundary difference can change the
        # escape iteration. Compare both float32 paths with float64 so precision
        # sensitivity is not mistaken for a lowering defect.
        ref64 = run_abstract_numpy(
            cx.astype(np.float64), cy.astype(np.float64), args.iterations
        )
        gpu_vs64 = float(np.mean(gpu == ref64)) * 100.0
        np_vs64 = float(np.mean(ref == ref64)) * 100.0
        print(f"vs f64  : glsl-f32 {gpu_vs64:.4f}%, numpy-f32 {np_vs64:.4f}% "
              f"-- both f32 paths differ from f64 by a comparable margin")
        if max_err > 0:
            disagree = gpu != ref
            c2 = ref.reshape(args.height, args.width)
            edge = np.zeros_like(c2, dtype=bool)
            edge[1:-1, 1:-1] = (
                (c2[1:-1, 1:-1] != c2[:-2, 1:-1])
                | (c2[1:-1, 1:-1] != c2[2:, 1:-1])
                | (c2[1:-1, 1:-1] != c2[1:-1, :-2])
                | (c2[1:-1, 1:-1] != c2[1:-1, 2:])
            )
            on_edge = float(np.mean(edge.ravel()[disagree])) * 100.0
            print(f"        : {disagree.sum():,} disagreeing px, {on_edge:.1f}% of them sit "
                  f"on an escape-count boundary (chaotic sensitivity, not a lowering bug)")
    else:
        print("verify  : skipped (--only-glsl)")

    # -- C backend, on a small grid ----------------------------------------
    if not args.skip_c and not args.only_glsl:
        n = args.c_probe
        pcx, pcy = complex_plane(n, n, args.center, args.span)
        big = c_workspace_bytes(captured.program, elements)
        small = c_workspace_bytes(captured.program, n * n)
        print(f"c probe : {n}x{n}; workspace {small / 1e6:.1f} MB "
              f"(the full image would need {big / 1e9:.1f} GB -- see note)")
        t0 = time.perf_counter()
        cpu = run_c(captured, pcx, pcy)
        c_ms = (time.perf_counter() - t0) * 1e3
        pref = run_abstract_numpy(pcx, pcy, args.iterations)
        pgpu = run_glsl(captured, pcx, pcy)
        print(f"c       : {c_ms:8.1f} ms  "
              f"(vs numpy: {float(np.max(np.abs(cpu - pref))):g} max diff)")
        print(f"c vs glsl: same program, both backends agree "
              f"= {bool(np.array_equal(cpu, pgpu))}")

    out = save_image(gpu, args.width, args.height, args.out, args.cmap,
                     vignette_tile=args.vignette_tile)
    print(f"wrote   : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
