"""The MICROSCOPE LENS: the first graph shader of SpeakToMe research case
bcdbf5dc13b94a628b7fdc3a9fab08ee.

Pure math over AbstractTensor; no I/O, no host state.  The function is the
source of truth: the numeric reference runs it directly on a numpy-backed
AbstractTensor, and the compiler lowers this same source to GLSL, WebGL and
LLVM (tools/build_graph_lens_shader.py).

Inputs (N nodes):
    P  [N, 2]  rest positions in layout space
    F  [2]     focus position
    H  [N]     graph-hop distance from the focus node(s), H >= 0
    k          lens strength, k > -1 (k = 0 is the identity)
    R          lens radius, R > 0
    d          depth-of-field scale, d >= 0

Outputs:
    Q  [N, 2]  displaced positions
    C  [N]     circle-of-confusion radius
    A  [N]     alpha

Formulas (u = r / R, r = |P - F|):
    Q = F + (P - F) * s(u)
    s(u) = (k + 1) / (k * min(u, 1) + 1)
    equivalently the Sarkar-Brown fisheye on the radius,
        r' = R (k + 1) (r / R) / (k (r / R) + 1)   for r < R,   r' = r   for r >= R
    (r' = r * s).  r' is strictly increasing in r (dr'/dr = (k+1)/(k u + 1)^2 > 0
    inside, 1 outside), so no two points are ever reordered radially; r' = R at
    r = R from both sides, so the map is continuous at the rim.  Near the focus
    the magnification is k + 1; toward the rim it falls to 1, compressing the
    context into the annulus.
    C = d * (sqrt(1 + H^2) - 1)        grows monotonically with H, C(0) = 0
    A = 1 / (1 + d * H)                falls monotonically with H, A(0) = 1
"""

from __future__ import annotations

from src.common.tensors.abstraction import AbstractTensor

#: Declared extents used by the build and by the tests.
NODE_COUNT = 64


def graph_lens(P: AbstractTensor, F: AbstractTensor, H: AbstractTensor,
               k: float, R: float, d: float):
    offset = P - F
    radius = ((offset * offset).sum(dim=1, keepdim=True)).sqrt()
    inside = (radius * (1.0 / R)).clamp_max(1.0)
    scale = (k + 1.0) / (inside * k + 1.0)
    Q = F + offset * scale
    C = ((H * H + 1.0).sqrt() - 1.0) * d
    A = 1.0 / (H * d + 1.0)
    return Q, C, A
