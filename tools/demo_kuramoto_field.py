"""A differential field of transcendentals: coupled phase oscillators.

Every cell of a torus holds one oscillator with its own natural frequency,
and each is pulled toward its four neighbours by the sine of their phase
difference::

    dtheta/dt = omega + K * sum_neighbours sin(theta_neighbour - theta)

That is the Kuramoto model on a lattice, and it is the cleanest thing this
tree could ask a transcendental pack to compute. The right-hand side IS a
transcendental -- four sines per cell per step, every one of them
independent of every other -- so a 256x256 field at sixty steps evaluates
sixteen million sines with no data dependence between them. Nothing else
in the field is more than add, subtract and multiply, which is exactly the
set ``Precision`` endorses, so the same program runs wide without a single
new operator.

WHY THIS IS WORTH COMPILING RATHER THAN CALLING. The mathematics is a
handful of lines and the interpreter is where they die: each step is a
separate dispatched tensor operation over the whole field, so the
arithmetic is a rounding error on the cost of reaching it. The point of
the pack is that the SAME authored source becomes a compiled kernel, and
the transcendental it leans on is ours -- measured against an exact
oracle, admitted or refused on the evidence -- rather than whatever libm
the platform happened to link.

WHAT THE FIELD DOES, so a reader knows whether the picture is right.
Identical oscillators synchronise into one phase. A spread of natural
frequencies fights that, and the balance between the spread and the
coupling K decides whether the field locks, breaks into domains, or --
in the interesting middle -- holds synchronised and desynchronised
regions at once, which is the chimera state Kuramoto and Battogtokh
found in 2002. The order parameter below measures which happened: it is
the length of the mean unit vector over all phases, one when the field
is locked and near zero when the phases are scattered.

Run::

    python -m tools.demo_kuramoto_field
    python -m tools.demo_kuramoto_field --width 128 --height 128 --steps 400
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_convolution.laplace_nd import Transform


#: The field advance, authored as ordinary Python over flat buffers.
#:
#: Flat buffers with computed indices, the loop bounds passed in, no hand
#: optimisation: the authoring rules the kernel pack already follows, so
#: this source can be ingested rather than reimplemented. The reference
#: this demo scores against is THIS TEXT executed as Python -- not a
#: second implementation, which would only answer whether two authors
#: agree.
FIELD_SOURCE = """
def kuramoto_step(theta, omega, out, width, height, coupling, dt):
    for row in range(height):
        for column in range(width):
            here = row * width + column
            up = ((row + height - 1) % height) * width + column
            down = ((row + 1) % height) * width + column
            left = row * width + (column + width - 1) % width
            right = row * width + (column + 1) % width
            phase = theta[here]
            pull = sin(theta[up] - phase)
            pull = pull + sin(theta[down] - phase)
            pull = pull + sin(theta[left] - phase)
            pull = pull + sin(theta[right] - phase)
            out[here] = phase + dt * (omega[here] + coupling * pull)
    return out
"""


def authored_step():
    """The advance, as a callable, from the source the compiler is given."""

    namespace = {"sin": math.sin}
    exec(compile(FIELD_SOURCE, "<kuramoto-field>", "exec"), namespace)
    return namespace["kuramoto_step"]


def vector_step(theta, omega, width, height, coupling, dt):
    """The same advance over whole arrays, for a field large enough to see.

    Identical mathematics to ``FIELD_SOURCE`` and deliberately kept beside
    it: the authored loop is what compiles, this is what makes a
    two-hundred-frame animation take seconds instead of an afternoon while
    the compiled lane is still being built. The two are checked against
    each other below rather than trusted.
    """

    grid = theta.reshape(height, width)
    pull = (
        np.sin(np.roll(grid, 1, axis=0) - grid)
        + np.sin(np.roll(grid, -1, axis=0) - grid)
        + np.sin(np.roll(grid, 1, axis=1) - grid)
        + np.sin(np.roll(grid, -1, axis=1) - grid)
    )
    return (grid + dt * (omega.reshape(height, width) + coupling * pull)).ravel()


# ---------------------------------------------------------------------------
# The same field on a curved surface.
#
# The coupling term is a LAPLACIAN in disguise: for small phase differences
# sin(d) is d, so summing sin(theta_neighbour - theta) is the discrete
# Laplacian of the phase, and the Kuramoto field is a heat equation whose
# conductivity saturates. That is the hinge this section turns on -- if the
# coupling is a Laplacian, then the repository's METRIC Laplacian says what
# the coupling becomes on a surface that is not flat.
#
# The geometry is not reimplemented here. A transform supplies an embedding,
# ``TransformHub.calculate_geometry`` returns the induced metric it implies
# (first fundamental form, its inverse, its determinant), and the coupling
# reads those weights. Curvature then steers the synchronisation the way it
# steers heat: waves run cheaply along directions the metric shortens and
# reluctantly across ones it stretches.
# ---------------------------------------------------------------------------


class BumpTransform(Transform):
    """A height field over the unit square: (u, v) -> (u, v, amplitude*bump).

    Deliberately the simplest embedding with non-constant curvature, so the
    metric it induces is easy to reason about and impossible to confuse with
    a coordinate artefact: the surface is flat at the edges and domed in the
    middle, and the dome is the only thing the coupling can be reacting to.
    """

    def __init__(self, amplitude: float = 0.6, device: str = "cpu"):
        self.amplitude = float(amplitude)
        self.uextent = self.vextent = 1.0
        self.wextent = 1.0
        self.device = device
        self.N_x = self.N_y = self.N_w = None
        self.u_mode = self.v_mode = self.w_mode = None
        self.grid_boundaries = (True,) * 6

    def get_transform_parameters(self):
        return (self.uextent, self.vextent, self.wextent), self.grid_boundaries

    def transform_spatial(self, grid_u, grid_v, grid_w):
        centred_u = (grid_u - 0.5) * 4.0
        centred_v = (grid_v - 0.5) * 4.0
        height = self.amplitude * (
            -(centred_u * centred_u + centred_v * centred_v)
        ).exp()
        return grid_u, grid_v, grid_w + height


def metric_weights(width: int, height: int, amplitude: float):
    """Per-direction coupling weights, read off the induced metric.

    Laplace-Beltrami on a surface is ``(1/sqrt(g)) d_i(sqrt(g) g^ij d_j f)``,
    so a nearest-neighbour discretisation weights the difference along each
    axis by ``sqrt(g) g^ii`` and divides the sum by ``sqrt(g)``.

    The metric is the FIRST FUNDAMENTAL FORM of the embedding, ``J^T J``,
    and the Jacobian comes from the repository's own transform machinery:
    ``compute_partials_and_normals`` differentiates the embedding through
    autograd and returns the nine partials grouped by parameter. Forming
    the metric from them here rather than asking ``calculate_geometry``
    for it is deliberate -- that path runs a ``metric_tensor_func`` whose
    default is an identity placeholder (the metric-steered demo supplies
    its own), so it reports a flat metric for a curved surface. The
    partials are the measured fact; the metric is their definition.
    """

    transform = BumpTransform(amplitude=amplitude)
    axis_u = np.linspace(0.0, 1.0, width, endpoint=False)
    axis_v = np.linspace(0.0, 1.0, height, endpoint=False)
    grid_u, grid_v, grid_w = np.meshgrid(
        axis_u, axis_v, np.zeros(1), indexing="xy",
    )
    partials = transform.compute_partials_and_normals(
        *[AbstractTensor.get_tensor(each) for each in
          (grid_u, grid_v, grid_w)]
    )

    def held(index):
        return np.asarray(partials[index].tolist())[..., 0]

    # partials[3:12] are (X_u, Y_u, Z_u, X_v, Y_v, Z_v, X_w, Y_w, Z_w).
    du = [held(3), held(4), held(5)]
    dv = [held(6), held(7), held(8)]
    g_uu = sum(component * component for component in du)
    g_vv = sum(component * component for component in dv)
    g_uv = sum(left * right for left, right in zip(du, dv))
    determinant = g_uu * g_vv - g_uv * g_uv
    root = np.sqrt(np.abs(determinant))
    # Inverse of the 2x2 form, whose diagonal is what a nearest-neighbour
    # stencil along each axis carries.
    return (
        root * g_vv / determinant,   # sqrt(g) * g^uu, along u
        root * g_uu / determinant,   # sqrt(g) * g^vv, along v
        root,
    )


def curved_step(theta, omega, weight_u, weight_v, root, width, height,
                coupling, dt):
    """One advance with the coupling steered by the surface's own metric."""

    grid = theta.reshape(height, width)
    face_u = 0.5 * (weight_u + np.roll(weight_u, -1, axis=1))
    face_v = 0.5 * (weight_v + np.roll(weight_v, -1, axis=0))
    pull = (
        np.roll(face_v, 1, axis=0) * np.sin(np.roll(grid, 1, axis=0) - grid)
        + face_v * np.sin(np.roll(grid, -1, axis=0) - grid)
        + np.roll(face_u, 1, axis=1) * np.sin(np.roll(grid, 1, axis=1) - grid)
        + face_u * np.sin(np.roll(grid, -1, axis=1) - grid)
    )
    advanced = grid + dt * (
        omega.reshape(height, width) + coupling * pull / root
    )
    return advanced.ravel()


def order_parameter(theta) -> float:
    """GLOBAL lock: the length of the mean unit phase vector.

    One when every oscillator in the field shares a phase, near zero when
    they are scattered. Reported because it is the classic Kuramoto
    number, but read it beside the local coherence below rather than
    alone -- see that function for why it is the wrong question here.
    """

    return float(np.hypot(np.mean(np.cos(theta)), np.mean(np.sin(theta))))


def local_coherence(theta, width: int, height: int) -> float:
    """LOCAL lock: how well each cell agrees with the neighbours it feels.

    The oscillators are coupled to their four neighbours and to nobody
    else, so the field locks into DOMAINS -- patches that each pick their
    own phase -- and the global order parameter, which averages over all
    of them, reads near zero however perfectly each domain has locked.
    Measured on a field that had visibly organised into smooth cells, the
    global number said 0.005 while every neighbour pair in it agreed to
    better than a tenth of a radian. Averaging the cosine of the phase
    difference across the EDGES asks the question the coupling actually
    answers: one when neighbours agree, zero when they are unrelated.
    """

    grid = theta.reshape(height, width)
    return float(np.mean([
        np.mean(np.cos(np.roll(grid, shift, axis=axis) - grid))
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1))
    ]))


def initial_field(width: int, height: int, spread: float, seed: int):
    """Random phases, and natural frequencies drawn once and held.

    The frequency spread is the whole experiment: at zero every
    oscillator wants the same rate and the field locks trivially, and
    wide enough that no coupling can hold them the field never locks at
    all. Between those is where the interesting states live.
    """

    generator = np.random.default_rng(seed)
    theta = generator.uniform(-math.pi, math.pi, width * height)
    omega = generator.normal(0.0, spread, width * height)
    return theta, omega


def agreement(width: int, height: int, spread: float, seed: int,
              coupling: float, dt: float, steps: int) -> float:
    """Worst disagreement between the authored loop and the vector form.

    The authored source is what will be compiled, so it is the one that
    has to be right; this checks that the fast path used for the frames
    computes the same field, on a small grid where running the explicit
    loop is affordable.
    """

    theta, omega = initial_field(width, height, spread, seed)
    step = authored_step()
    looped = np.array(theta)
    vectored = np.array(theta)
    scratch = np.zeros_like(theta)
    worst = 0.0
    for _index in range(steps):
        looped = np.array(step(
            list(looped), list(omega), list(scratch),
            width, height, coupling, dt,
        ))
        vectored = vector_step(vectored, omega, width, height, coupling, dt)
        worst = max(worst, float(np.max(np.abs(looped - vectored))))
    return worst


def render(theta, width: int, height: int):
    """Phase as hue: the picture the field is for.

    Phase is an angle, so it must be shown by something that wraps --
    a linear colour ramp would draw a false seam where the phase passes
    through pi and back, exactly where the field is most continuous.
    """

    grid = theta.reshape(height, width)
    wrapped = np.mod(grid, 2.0 * math.pi) / (2.0 * math.pi)
    red = 0.5 + 0.5 * np.cos(2.0 * math.pi * wrapped)
    green = 0.5 + 0.5 * np.cos(2.0 * math.pi * (wrapped - 1.0 / 3.0))
    blue = 0.5 + 0.5 * np.cos(2.0 * math.pi * (wrapped - 2.0 / 3.0))
    return np.clip(np.stack([red, green, blue], axis=-1), 0.0, 1.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--coupling", type=float, default=0.35)
    parser.add_argument("--spread", type=float, default=0.55)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--curvature", type=float, default=0.6,
        help="dome height of the embedded surface; 0 is the flat torus",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("build/kuramoto-field"),
    )
    arguments = parser.parse_args(argv)

    width, height = int(arguments.width), int(arguments.height)
    print(
        f"field {width}x{height}, coupling {arguments.coupling}, "
        f"frequency spread {arguments.spread}, dt {arguments.dt}",
        flush=True,
    )

    # The authored source is the thing that compiles, so it is checked
    # first and on its own terms, before any picture is drawn from the
    # faster path.
    worst = agreement(16, 16, arguments.spread, arguments.seed,
                      arguments.coupling, arguments.dt, 8)
    print(
        f"authored loop vs vector form, 16x16 over 8 steps: "
        f"worst |difference| = {worst:.3e}",
        flush=True,
    )

    theta, omega = initial_field(width, height, arguments.spread,
                                 arguments.seed)

    # The same initial field advanced two ways: on the flat torus, and on
    # a surface whose metric steers the coupling. Same phases, same
    # frequencies, same coupling constant -- so any difference between
    # the two pictures is the geometry and nothing else.
    weight_u, weight_v, root = metric_weights(width, height,
                                              arguments.curvature)
    print(
        f"metric weights on the dome: {weight_u.min():.3f}..{weight_u.max():.3f}"
        f"  sqrt(det g): {root.min():.3f}..{root.max():.3f}",
        flush=True,
    )
    print()
    print(
        f"{'step':>6}  {'flat global':>11}  {'flat local':>10}  "
        f"{'curved local':>12}  {'sines':>14}",
        flush=True,
    )
    started = time.perf_counter()
    sines = 0
    flat, curved = np.array(theta), np.array(theta)
    flat_frames, curved_frames = [], []
    interval = max(1, int(arguments.steps) // 5)
    for step_index in range(int(arguments.steps) + 1):
        if step_index % interval == 0:
            print(
                f"{step_index:6d}  {order_parameter(flat):11.4f}  "
                f"{local_coherence(flat, width, height):10.4f}  "
                f"{local_coherence(curved, width, height):12.4f}  "
                f"{sines:14,d}",
                flush=True,
            )
            flat_frames.append(render(flat, width, height))
            curved_frames.append(render(curved, width, height))
        flat = vector_step(flat, omega, width, height,
                           arguments.coupling, arguments.dt)
        curved = curved_step(curved, omega, weight_u, weight_v, root,
                             width, height, arguments.coupling, arguments.dt)
        sines += 8 * width * height
    elapsed = time.perf_counter() - started

    print()
    print(
        f"{sines:,d} sine evaluations in {elapsed:.2f}s "
        f"({sines / max(elapsed, 1e-9) / 1e6:.1f} million/second, "
        "numpy's libm)",
        flush=True,
    )
    print(
        "every one of them independent within a step: this is the work "
        "the compiled pack exists to take over",
        flush=True,
    )

    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        figure, axes = pyplot.subplots(2, len(flat_frames), figsize=(
            2.6 * len(flat_frames), 5.6,
        ))
        for column, (upper, lower) in enumerate(
            zip(flat_frames, curved_frames)
        ):
            axes[0][column].imshow(upper, interpolation="nearest")
            axes[0][column].set_title(f"step {column * interval}")
            axes[1][column].imshow(lower, interpolation="nearest")
            for row in (0, 1):
                axes[row][column].axis("off")
        axes[0][0].set_ylabel("flat")
        axes[1][0].set_ylabel("curved")
        figure.suptitle(
            "Kuramoto field, phase as hue -- top: flat torus, "
            f"bottom: metric-steered dome (K={arguments.coupling}, "
            f"spread={arguments.spread}, curvature={arguments.curvature})"
        )
        figure.tight_layout()
        path = destination / "kuramoto_field.png"
        figure.savefig(path, dpi=110)
        pyplot.close(figure)
        print(f"wrote {path}", flush=True)
    except ImportError:
        print("matplotlib is absent; skipped the picture", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
