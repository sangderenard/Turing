import argparse
from typing import Sequence, Tuple, Dict, Any, Optional

import numpy as np
import logging

from src.cells.cellsim.data.state import Cell, Bath
from src.cells.cellsim.api.saline import SalinePressureAPI
from src.cells.cellsim.mechanics.softbody0d import SoftbodyProviderCfg

# Lightweight math helpers (duplicated to avoid importing OpenGL demo)
import math

logger = logging.getLogger(__name__)

def _perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m

def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = center - eye; f = f / (np.linalg.norm(f) + 1e-12)
    s = np.cross(f, up); s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[:3, 3] = -np.array([np.dot(s, eye), np.dot(u, eye), np.dot(-f, eye)], dtype=np.float32)
    return m

def _translate(tvec: np.ndarray) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[0, 3] = float(tvec[0])
    m[1, 3] = float(tvec[1])
    m[2, 3] = float(tvec[2])
    return m

def _rotate_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = c; m[0, 2] = s
    m[2, 0] = -s; m[2, 2] = c
    return m


def make_cellsim_backend(*,
    cell_vols: Sequence[float],
    cell_imps: Sequence[float],
    cell_elastic_k: Sequence[float],
    bath_na: float,
    bath_cl: float,
    bath_pressure: float,
    bath_volume_factor: float,
    substeps: int,
    dt_provider: float,
    dim: int = 3,
):
    """Build a cellsim system attached to the softbody 0D provider.

    Returns (api, provider).
    """
    if not (len(cell_vols) == len(cell_imps) == len(cell_elastic_k)):
        raise ValueError("cell parameters must have the same length")

    cells = []
    for V, imp, k in zip(cell_vols, cell_imps, cell_elastic_k):
        cells.append(
            Cell(
                V=float(V),
                n={"Imp": float(imp), "Na": 0.0, "K": 0.0, "Cl": 0.0},
                elastic_k=float(k),
            )
        )

    bath = Bath(
        V=sum(cell_vols) * bath_volume_factor,
        n={"Na": float(bath_na), "K": 0.0, "Cl": float(bath_cl), "Imp": 0.0},
        pressure=float(bath_pressure),
    )

    api = SalinePressureAPI(cells, bath)
    provider = api.attach_softbody_mechanics(
        SoftbodyProviderCfg(substeps=substeps, dt_provider=dt_provider, dim=dim)
    )
    return api, provider


def step_cellsim(api: SalinePressureAPI, dt: float) -> float:
    """Advance cellsim one step and finalize bath thermodynamics.

    The Saline engine is stepped first; the attached :class:`~transmogrifier.cells.bath.fluid.Bath`
    layer then exposes its latest pressure/temperature/viscosity for downstream
    diagnostics.  The returned value is the engine's suggested next ``dt`` and
    the bath diagnostics are stored on ``api.last_bath_state``.
    """

    dt = api.step(dt)
    # Bath diagnostics for viewers (robust to missing attributes)
    bath = getattr(api, "bath", None)
    if bath is not None and hasattr(bath, "finalize_step"):
        try:
            api.last_bath_state = bath.finalize_step()
            logger.debug("Bath state: %s", api.last_bath_state)
        except Exception:
            api.last_bath_state = None
            logger.exception("Bath finalization failed")
    else:
        api.last_bath_state = None
    return dt


def _com_and_com_vel(cell):
    """Compute COM position and COM velocity for a softbody cell.

    Returns (com: np.ndarray shape (3,), vcom: np.ndarray shape (3,)).
    Uses mass weighting from inverse masses (ignores pinned verts where invm==0).
    """
    invm = getattr(cell, "invm", None)
    X = getattr(cell, "X", None)
    V = getattr(cell, "V", None)
    if invm is None or X is None or V is None:
        # Fallback to zeros if structure is unexpected
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    m = np.where(invm > 0, 1.0 / invm, 0.0)
    if m.sum() == 0:
        w = np.full(len(invm), 1.0 / max(1, len(invm)))
    else:
        w = m / m.sum()
    com = (X * w[:, None]).sum(axis=0)
    vcom = (V * w[:, None]).sum(axis=0)
    return com, vcom


def build_numpy_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Create an ArgumentParser with shared sim parameters for all demos.

    Use add_help=False when composing this as a parent parser in other demos.
    """
    parser = argparse.ArgumentParser(
        description="Run softbody cellsim with numpy-only backend",
        add_help=add_help,
    )
    parser.add_argument("--cell-vols", type=float, nargs="+", default=[10.0, 10.0, 10.0])
    parser.add_argument("--cell-imps", type=float, nargs="+", default=[100.0, 100.0, 100.0])
    parser.add_argument("--cell-elastic-k", type=float, nargs="+", default=[0.1, 0.1, 0.1])
    parser.add_argument("--bath-volume-factor", type=float, default=5.0)
    parser.add_argument("--bath-na", type=float, default=10.0)
    parser.add_argument("--bath-cl", type=float, default=10.0)
    parser.add_argument("--bath-pressure", type=float, default=1e1)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument(
        "--dt-provider", type=float, default=1e-10,
        help="internal softbody timestep; scale to speed up motion",
    )
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument(
        "--dt", type=float, default=1e-10,
        help="base integrator step; increase to amplify drift",
    )
    parser.add_argument(
        "--fluid", choices=["", "discrete", "voxel"], default="",
        help="Run stand-alone fluid demo (discrete or voxel) instead of cellsim",
    )
    # Export/stream options
    parser.add_argument("--export-npz", type=str, default="",
                        help="If set, write a prerendered NPZ stream to this path.")
    parser.add_argument("--export-kind", choices=["ascii", "opengl-points", "opengl-mesh"], default="",
                        help="What kind of stream to export when --export-npz is set.")
    parser.add_argument("--stream", choices=["", "ascii", "opengl-points", "opengl-mesh"], default="",
                        help="If set, stream frames live to the chosen visualizer in-process (no files).")
    parser.add_argument("--loop", choices=["none", "loop", "bounce"], default="none",
                        help="Playback mode for streaming/NPZ viewers.")
    parser.add_argument("--fps", type=float, default=60.0, help="Target FPS for streaming display.")
    parser.add_argument("--ascii-nx", type=int, default=120, help="ASCII export grid width")
    parser.add_argument("--ascii-ny", type=int, default=36, help="ASCII export grid height")
    parser.add_argument("--render-mode", choices=["edges", "fill"], default="fill",
                        help="ASCII export render mode (edges/fill)")
    parser.add_argument("--face-stride", type=int, default=1,
                        help="ASCII export: draw every Nth face (performance)")
    parser.add_argument("--no-points", action="store_true",
                        help="ASCII export: disable drawing vertex points")
    parser.add_argument("--gl-fovy", type=float, default=45.0, help="OpenGL stream: vertical FOV degrees")
    parser.add_argument("--gl-rot-speed", type=float, default=0.25, help="OpenGL stream: Y-rotation speed rad/sec (used as frame*dt)")
    parser.add_argument("--gl-viewport-w", type=int, default=1100, help="OpenGL stream: viewport width")
    parser.add_argument("--gl-viewport-h", type=int, default=800, help="OpenGL stream: viewport height")
    parser.add_argument("--show-vectors", action="store_true",
                        help="OpenGL stream: render velocity vectors as arrows")
    parser.add_argument("--color-metric", choices=["none", "magnitude"], default="magnitude",
                        help="Metric for arrow coloration (none/magnitude)")
    parser.add_argument("--arrow-scale", type=float, default=1.0,
                        help="Scale factor applied to arrow lengths")
    parser.add_argument("--flow-anim-speed", type=float, default=1.0,
                        help="Multiplier for flow animation speed in viewers")
    parser.add_argument("--verbose", action="store_true", help="Log per-cell parameters each frame")
    parser.add_argument("--debug", action="store_true", help="Log full per-vertex and per-face data")
    parser.add_argument("--sim-dim", type=int, choices=[1, 2, 3], default=3,
                        help="Softbody simulation dimension")
    return parser


def parse_args():
    return build_numpy_parser(add_help=True).parse_args()


def _gather_vertices(h) -> Optional[np.ndarray]:
    cells = getattr(h, 'cells', [])
    if not cells:
        return None
    try:
        allX = np.concatenate([c.X for c in cells], axis=0).astype(np.float32)
        return allX
    except Exception:
        return None

def _cells_faces_counts(h) -> Tuple[np.ndarray, np.ndarray]:
    """Return (faces_concat uint32 flat, faces_counts per cell)."""
    faces_list = []
    counts = []
    for c in getattr(h, 'cells', []) or []:
        F = np.asarray(c.faces, dtype=np.uint32).reshape(-1, 3)
        faces_list.append(F)
        counts.append(F.shape[0])
    if not faces_list:
        return np.zeros((0,), dtype=np.uint32), np.zeros((0,), dtype=np.int32)
    faces_concat = np.concatenate(faces_list, axis=0).astype(np.uint32).ravel()
    return faces_concat, np.array(counts, dtype=np.int32)

def _cells_vertex_counts(h) -> np.ndarray:
    return np.array([len(np.asarray(c.X)) for c in getattr(h, 'cells', [])], dtype=np.int32)


def _array_stats(arr: np.ndarray) -> Tuple[Any, Any, Any]:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.min().tolist(), arr.max().tolist(), arr.mean().tolist()
    return (
        arr.min(axis=0).tolist(),
        arr.max(axis=0).tolist(),
        arr.mean(axis=0).tolist(),
    )


def _log_hierarchy_state(h, frame: int, api: SalinePressureAPI | None = None, *, debug: bool = False) -> None:
    if h is None:
        logger.info("frame %d: <no hierarchy>", frame)
        return
    if api is not None:
        bstate = getattr(api, "last_bath_state", None)
        if isinstance(bstate, dict):
            logger.info(
                "frame %d bath pressure=%s temperature=%s viscosity=%s",
                frame,
                bstate.get("pressure"),
                bstate.get("temperature"),
                bstate.get("viscosity"),
            )
    for ci, c in enumerate(getattr(h, 'cells', []) or []):
        logger.info("frame %d cell %d", frame, ci)
        for attr, val in c.__dict__.items():
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    logger.info("  %s: shape=%s", attr, val.shape)
                else:
                    mn, mx, mean = _array_stats(val)
                    logger.info(
                        "  %s: shape=%s min=%s max=%s mean=%s",
                        attr,
                        val.shape,
                        mn,
                        mx,
                        mean,
                    )
                    if debug:
                        logger.debug("  %s values=%s", attr, val.tolist())
            else:
                logger.info("  %s=%s", attr, val)

def _compute_center_radius(h) -> Tuple[np.ndarray, float]:
    allX = np.concatenate([c.X for c in h.cells], axis=0).astype(np.float32)
    bmin, bmax = allX.min(0), allX.max(0)
    center = (bmin + bmax) * 0.5
    radius = 0.5 * float(np.linalg.norm(bmax - bmin))
    return center.astype(np.float32), radius


def _compute_center_radius_pts(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    if pts.size == 0:
        return np.zeros(3, dtype=np.float32), 1.0
    pts = pts.astype(np.float32, copy=False)
    bmin, bmax = pts.min(0), pts.max(0)
    center = (bmin + bmax) * 0.5
    radius = 0.5 * float(np.linalg.norm(bmax - bmin))
    return center.astype(np.float32), radius

def _measure_pressure_mass(api, h) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pressures, masses, greens = [], [], []
    for i, c in enumerate(getattr(h, 'cells', []) or []):
        p = 0.0
        if hasattr(c, "contact_pressure_estimate"):
            try:
                p = float(c.contact_pressure_estimate())
            except Exception:
                p = 0.0
        elif i < len(getattr(api, 'cells', [])):
            p = float(getattr(api.cells[i], "internal_pressure", 0.0))
        n = getattr(api.cells[i], "n", None) if i < len(getattr(api, 'cells', [])) else None
        masses.append(sum(n.values()) if isinstance(n, dict) else 0.0)
        g = getattr(api.cells[i], "_identity_green", getattr(c, "_identity_green", 128)) if i < len(getattr(api, 'cells', [])) else getattr(c, "_identity_green", 128)
        greens.append(int(g))
        pressures.append(p)
    return np.array(pressures), np.array(masses), np.array(greens, dtype=np.int32)

def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x)) if x.size else 0.0
    return (x / m) if m > 1e-12 else np.zeros_like(x, dtype=np.float64)

def _ascii_colors_per_cell(api, h) -> np.ndarray:
    """Compute per-cell RGB color as float32 in [0,1], using pressure (B), mass (R), identity green (G)."""
    pressures, masses, greens255 = _measure_pressure_mass(api, h)
    pN = _normalize(pressures)
    mN = _normalize(masses)
    BASE_R, BASE_B = 0.15, 0.25
    PRESSURE_GAIN, MASS_GAIN = 0.75, 0.75
    n = len(getattr(h, 'cells', []) or [])
    cols = np.zeros((n, 3), dtype=np.float32)
    if n == 0:
        return cols
    Gs = (greens255.astype(np.float32) / 255.0)
    R = np.minimum(1.0, BASE_R + MASS_GAIN * mN.astype(np.float32))
    B = np.minimum(1.0, BASE_B + PRESSURE_GAIN * pN.astype(np.float32))
    cols[:, 0] = R
    cols[:, 1] = Gs
    cols[:, 2] = B
    return cols

def _rasterize_ascii_numpy(h, api, nx: int, ny: int, *, render_mode: str, face_stride: int, draw_points: bool):
    """Rasterize meshes to ASCII buffers using only NumPy arrays (no Pythonic grid structures).

    Returns (chars uint8 [ny,nx], rgb uint8 [ny,nx,3]).
    """
    nx = int(nx); ny = int(ny)
    chars = np.full((ny, nx), ord(' '), dtype=np.uint8)
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)

    cells = getattr(h, 'cells', []) or []
    if not cells:
        return chars, rgb

    # Per-cell color (float 0..1)
    cols = _ascii_colors_per_cell(api, h)
    # Default identity greens if missing
    if cols.shape[0] != len(cells):
        cols = np.tile(np.array([[0.3, 0.5, 0.4]], dtype=np.float32), (len(cells), 1))

    stride = max(1, int(face_stride))

    def draw_line_ixy(ix0: int, iy0: int, ix1: int, iy1: int, color_u8: np.ndarray, ch_code: int):
        # Bresenham directly into numpy arrays
        x0 = int(ix0); y0 = int(iy0); x1 = int(ix1); y1 = int(iy1)
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            if 0 <= x < nx and 0 <= y < ny:
                chars[y, x] = ch_code
                rgb[y, x, :] = color_u8
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    for ci, c in enumerate(cells):
        X = np.asarray(getattr(c, 'X', None))
        F = np.asarray(getattr(c, 'faces', None))
        if X is None or X.ndim != 2 or X.shape[1] < 2:
            continue
        has_faces = F.ndim == 2 and F.size > 0
        # Project XY to grid indices (vectorized)
        ix = np.clip((X[:, 0] * nx).astype(np.int32), 0, nx - 1)
        iy = np.clip((X[:, 1] * ny).astype(np.int32), 0, ny - 1)

        # Convert color to u8 once per cell
        col = cols[ci]
        col_u8 = np.array([int(col[0] * 255.0), int(col[1] * 255.0), int(col[2] * 255.0)], dtype=np.uint8)

        if draw_points:
            # Subsample points based on density (vectorized indexing)
            k = max(1, int(len(X) / (nx * 0.75)))
            sel = np.arange(0, len(X), k, dtype=np.int32)
            px = ix[sel]; py = iy[sel]
            ch = ord('.')
            mask = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
            chars[py[mask], px[mask]] = ch
            rgb[py[mask], px[mask], :] = col_u8

        if render_mode in ("edges", "fill") and has_faces:
            # Support triangular faces (shape (?,3)) and polylines (shape (?,2))
            if F.shape[1] == 2:
                E = F.astype(np.int32)[::stride]
            else:
                F = F.reshape(-1, 3)[::stride]
                e01 = F[:, [0, 1]]; e12 = F[:, [1, 2]]; e20 = F[:, [2, 0]]
                E = np.concatenate([e01, e12, e20], axis=0)
            # Remove degenerate edges
            valid = (E[:, 0] != E[:, 1])
            E = E[valid]
            if E.size:
                # Sample S points per edge
                S = 12  # samples per edge; adjust for thickness/continuity
                t = np.linspace(0.0, 1.0, S, dtype=np.float32)[None, :]
                x0 = X[E[:, 0], 0][:, None]; y0 = X[E[:, 0], 1][:, None]
                x1 = X[E[:, 1], 0][:, None]; y1 = X[E[:, 1], 1][:, None]
                xs = x0 + (x1 - x0) * t
                ys = y0 + (y1 - y0) * t
                ixs = np.clip((xs * nx).astype(np.int32), 0, nx - 1)
                iys = np.clip((ys * ny).astype(np.int32), 0, ny - 1)
                # Flatten and write
                flat_ix = ixs.reshape(-1)
                flat_iy = iys.reshape(-1)
                chars[flat_iy, flat_ix] = ord('#')
                rgb[flat_iy, flat_ix, :] = col_u8

        # Organelles as discs
        organs = getattr(c, 'organelles', []) or []
        if organs:
            # We will loop; still only manipulating numpy buffers
            for o in organs:
                cx = int(np.clip(int(o.pos[0] * nx), 0, nx - 1))
                cy = int(np.clip(int(o.pos[1] * ny), 0, ny - 1))
                pr = max(1, int(o.radius * max(nx, ny)))
                rr = pr * pr
                y0 = max(0, cy - pr); y1 = min(ny - 1, cy + pr)
                x0 = max(0, cx - pr); x1 = min(nx - 1, cx + pr)
                for yy in range(y0, y1 + 1):
                    dy = yy - cy
                    # horizontal span per y
                    dx_max = int((rr - dy * dy) ** 0.5) if rr >= dy * dy else 0
                    xs = max(x0, cx - dx_max); xe = min(x1, cx + dx_max)
                    if xe >= xs:
                        chars[yy, xs:xe + 1] = ord('o')
                        rgb[yy, xs:xe + 1, :] = col_u8

    return chars, rgb

def export_ascii_stream(args, api, provider):
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    nx, ny = int(args.ascii_nx), int(args.ascii_ny)
    chars = np.zeros((frames, ny, nx), dtype=np.uint8)
    rgb = np.zeros((frames, ny, nx, 3), dtype=np.uint8)

    for f in range(frames):
        dt = step_cellsim(api, dt)
        h = getattr(provider, "_h", None)
        if args.verbose or args.debug:
            _log_hierarchy_state(h, f, api=api, debug=args.debug)
        if h is None:
            continue
        ch, col = _rasterize_ascii_numpy(
            h, api, nx, ny,
            render_mode=args.render_mode,
            face_stride=args.face_stride,
            draw_points=not args.no_points,
        )
        chars[f] = ch
        rgb[f] = col

    meta = dict(stream_type='ascii_v1', nx=nx, ny=ny, frames=frames)
    np.savez_compressed(args.export_npz, **meta, chars=chars, rgb=rgb)

def export_opengl_points_stream(args, api, provider):
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    w, h = int(args.gl_viewport_w), int(args.gl_viewport_h)
    fovy = float(args.gl_fovy)
    rot_speed = float(args.gl_rot_speed)
    dim = getattr(getattr(provider, 'cfg', None), 'dim', 3)

    pts_offsets = np.zeros(frames + 1, dtype=np.int64)
    pts_concat_list = []
    vec_concat_list = []
    mvps = np.zeros((frames, 4, 4), dtype=np.float32)

    # Camera state similar to GL demo
    # Initialize once after a prime step so hierarchy exists
    api.step(1e-3)
    hobj = getattr(provider, "_h", None)
    if hobj is None:
        raise RuntimeError("Softbody provider failed to initialize _h")
    # initial camera
    center, radius = _compute_center_radius(hobj)
    eye = np.array([0.5, 0.5, 1.7], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam_dir = center - eye
    cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
    cam_dist = float(np.linalg.norm(center - eye))
    center_s = center.copy(); dist_s = cam_dist
    aspect = w / max(1, h)

    t_sim = 0.0
    for f in range(frames):
        dt = step_cellsim(api, dt)
        t_sim += dt
        hobj = getattr(provider, "_h", hobj)
        # gather points
        vtx = _gather_vertices(hobj)
        if vtx is None:
            vtx = np.zeros((0, 3), dtype=np.float32)
        pts_concat_list.append(vtx.astype(np.float32, copy=False))
        pts_offsets[f + 1] = pts_offsets[f] + int(len(vtx))

        # camera auto-framing & rotation
        new_center, radius = _compute_center_radius(hobj)
        desired_dist = max(0.2, radius / math.tan(math.radians(fovy * 0.5)) * 2.0)
        alpha = 0.15
        center_s = (1.0 - alpha) * center_s + alpha * new_center
        dist_s = (1.0 - alpha) * dist_s + alpha * desired_dist
        center = center_s; cam_dist = float(dist_s)
        eye = center - cam_dir * cam_dist

        P = _perspective(fovy, aspect, 0.05, max(10.0, cam_dist + 3.0 * radius))
        V = _look_at(eye, center, up)
        theta = rot_speed * t_sim if dim == 3 else 0.0  # no rotation in 1D/2D
        T_neg = _translate(-center)
        R_y = _rotate_y(theta)
        T_pos = _translate(center)
        M = (T_pos @ R_y @ T_neg).astype(np.float32)
        mvps[f] = (P @ V @ M).astype(np.float32)

    pts_concat = np.concatenate(pts_concat_list, axis=0) if pts_concat_list else np.zeros((0, 3), dtype=np.float32)
    meta = dict(stream_type='opengl_points_v1', frames=frames, viewport_w=w, viewport_h=h)
    np.savez_compressed(args.export_npz, **meta, pts_offsets=pts_offsets, pts_concat=pts_concat, mvps=mvps)

def export_opengl_mesh_stream(args, api, provider):
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    w, h = int(args.gl_viewport_w), int(args.gl_viewport_h)
    fovy = float(args.gl_fovy)
    rot_speed = float(args.gl_rot_speed)
    dim = getattr(getattr(provider, 'cfg', None), 'dim', 3)

    # Prime hierarchy
    api.step(1e-3)
    hobj = getattr(provider, "_h", None)
    if hobj is None:
        raise RuntimeError("Softbody provider failed to initialize _h")

    n_cells = len(getattr(hobj, 'cells', []) or [])
    vtx_counts = _cells_vertex_counts(hobj)
    faces_concat, face_counts = _cells_faces_counts(hobj)
    vtx_offsets = np.zeros(n_cells + 1, dtype=np.int64)
    for i in range(n_cells):
        vtx_offsets[i + 1] = vtx_offsets[i] + int(vtx_counts[i])

    mvps = np.zeros((frames, 4, 4), dtype=np.float32)
    colors = np.zeros((frames, n_cells, 4), dtype=np.float32)
    draw_order = np.zeros((frames, n_cells), dtype=np.int32)
    vtx_concat = np.zeros((frames, int(vtx_offsets[-1]), 3), dtype=np.float32)

    # Camera state
    center, radius = _compute_center_radius(hobj)
    eye = np.array([0.5, 0.5, 1.7], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam_dir = center - eye; cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
    cam_dist = float(np.linalg.norm(center - eye))
    center_s = center.copy(); dist_s = cam_dist
    aspect = w / max(1, h)

    t_sim = 0.0
    for f in range(frames):
        dt = step_cellsim(api, dt)
        t_sim += dt
        hobj = getattr(provider, "_h", hobj)
    # fill vertex positions into concat buffer (vectorized via concatenate)
    vtx_concat[f, :, :] = np.concatenate([np.asarray(c.X, dtype=np.float32) for c in hobj.cells], axis=0)

    # colors from pressures/masses
    pressures, masses, greens255 = _measure_pressure_mass(api, hobj)
    pN = _normalize(pressures)
    mN = _normalize(masses)
    BASE_R, BASE_B, BASE_A = 0.15, 0.25, 0.35
    PRESSURE_GAIN, MASS_GAIN = 0.75, 0.75
    G = (greens255.astype(np.float32) / 255.0)
    R = np.minimum(1.0, BASE_R + MASS_GAIN * mN.astype(np.float32))
    B = np.minimum(1.0, BASE_B + PRESSURE_GAIN * pN.astype(np.float32))
    colors[f, :, 0] = R
    colors[f, :, 1] = G
    colors[f, :, 2] = B
    colors[f, :, 3] = BASE_A

    # camera
    new_center, radius = _compute_center_radius(hobj)
    desired_dist = max(0.2, radius / math.tan(math.radians(fovy * 0.5)) * 2.0)
    alpha = 0.15
    center_s = (1.0 - alpha) * center_s + alpha * new_center
    dist_s = (1.0 - alpha) * dist_s + alpha * desired_dist
    center = center_s; cam_dist = float(dist_s)
    eye = center - cam_dir * cam_dist
    P = _perspective(fovy, aspect, 0.05, max(10.0, cam_dist + 3.0 * radius))
    V = _look_at(eye, center, up)
    theta = rot_speed * t_sim if dim == 3 else 0.0
    T_neg = _translate(-center)
    R_y = _rotate_y(theta)
    T_pos = _translate(center)
    M = (T_pos @ R_y @ T_neg).astype(np.float32)
    MVP = (P @ V @ M).astype(np.float32)
    mvps[f] = MVP

    # depth sort order using centroid depth after model rotation (vectorized)
    view_dir = (center - eye); view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)
    # Compute centroids per cell from concatenated vertices
    Xcat = vtx_concat[f]
    # segment sums using reduceat
    off = vtx_offsets.astype(np.int32)
    sums = np.add.reduceat(Xcat, off[:-1], axis=0)
    counts = (off[1:] - off[:-1]).astype(np.float32)[:, None]
    centroids = sums / np.maximum(counts, 1e-12)
    # rotate centroids
    centroids_h = np.concatenate([centroids, np.ones((n_cells, 1), dtype=np.float32)], axis=1)
    ctr_rot = (M @ centroids_h.T).T[:, :3]
    depths = np.dot(ctr_rot - eye[None, :], view_dir)
    draw_order[f, :] = np.argsort(depths).astype(np.int32)

    meta = dict(stream_type='opengl_mesh_v1', frames=frames, viewport_w=w, viewport_h=h)
    np.savez_compressed(
        args.export_npz,
        **meta,
        n_cells=n_cells,
        vtx_counts=vtx_counts,
        vtx_offsets=vtx_offsets,
        faces_concat=faces_concat,
        face_counts=face_counts,
        vtx_concat=vtx_concat,
        colors=colors,
        draw_order=draw_order,
        mvps=mvps,
    )

def stream_ascii(args, api, provider):
    # Build frames into NumPy arrays (no Pythonic grid) and stream directly
    from .run_ascii_demo import play_ascii_stream
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    nx, ny = int(args.ascii_nx), int(args.ascii_ny)
    chars = np.zeros((frames, ny, nx), dtype=np.uint8)
    rgb = np.zeros((frames, ny, nx, 3), dtype=np.uint8)

    for f in range(frames):
        dt = step_cellsim(api, dt)
        h = getattr(provider, "_h", None)
        if args.verbose or args.debug:
            _log_hierarchy_state(h, f, api=api, debug=args.debug)
        if h is None:
            continue
        ch, col = _rasterize_ascii_numpy(
            h, api, nx, ny,
            render_mode=args.render_mode,
            face_stride=args.face_stride,
            draw_points=not args.no_points,
        )
        chars[f] = ch
        rgb[f] = col

    play_ascii_stream(chars, rgb, color_mode="auto", loop_mode=args.loop, fps=float(args.fps))

def stream_opengl_points(args, api, provider):
    from .run_opengl_demo import play_points_stream
    # Build stream arrays (same as export, but feed player directly)
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    w, h = int(args.gl_viewport_w), int(args.gl_viewport_h)
    fovy = float(args.gl_fovy)
    rot_speed = float(args.gl_rot_speed)
    dim = getattr(getattr(provider, 'cfg', None), 'dim', 3)

    pts_offsets = np.zeros(frames + 1, dtype=np.int64)
    pts_concat_list = []
    vec_concat_list = []
    mvps = np.zeros((frames, 4, 4), dtype=np.float32)

    api.step(1e-3)
    hobj = getattr(provider, "_h", None)
    if hobj is None:
        raise RuntimeError("Softbody provider failed to initialize _h")

    center, radius = _compute_center_radius(hobj)
    eye = np.array([0.5, 0.5, 1.7], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam_dir = center - eye
    cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
    cam_dist = float(np.linalg.norm(center - eye))
    center_s = center.copy(); dist_s = cam_dist
    aspect = w / max(1, h)

    t_sim = 0.0
    for f in range(frames):
        dt = step_cellsim(api, dt)
        t_sim += dt
        hobj = getattr(provider, "_h", hobj)
        if args.verbose or args.debug:
            _log_hierarchy_state(hobj, f, api=api, debug=args.debug)
        vtx = _gather_vertices(hobj)
        if vtx is None:
            vtx = np.zeros((0, 3), dtype=np.float32)
        pts_concat_list.append(vtx.astype(np.float32, copy=False))
        pts_offsets[f + 1] = pts_offsets[f] + int(len(vtx))

        new_center, radius = _compute_center_radius(hobj)
        desired_dist = max(0.2, radius / math.tan(math.radians(fovy * 0.5)) * 2.0)
        alpha = 0.15
        center_s = (1.0 - alpha) * center_s + alpha * new_center
        dist_s = (1.0 - alpha) * dist_s + alpha * desired_dist
        center = center_s; cam_dist = float(dist_s)
        eye = center - cam_dir * cam_dist

        P = _perspective(fovy, aspect, 0.05, max(10.0, cam_dist + 3.0 * radius))
        V = _look_at(eye, center, up)
        theta = rot_speed * t_sim if dim == 3 else 0.0
        T_neg = _translate(-center)
        R_y = _rotate_y(theta)
        T_pos = _translate(center)
        M = (T_pos @ R_y @ T_neg).astype(np.float32)
        mvps[f] = (P @ V @ M).astype(np.float32)

    pts_concat = np.concatenate(pts_concat_list, axis=0) if pts_concat_list else np.zeros((0, 3), dtype=np.float32)
    play_points_stream(pts_offsets, pts_concat, mvps, viewport_w=w, viewport_h=h, loop_mode=args.loop, fps=float(args.fps))

def stream_opengl_mesh(args, api, provider):
    from .run_opengl_demo import play_mesh_stream
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    w, h = int(args.gl_viewport_w), int(args.gl_viewport_h)
    fovy = float(args.gl_fovy)
    rot_speed = float(args.gl_rot_speed)
    dim = getattr(getattr(provider, 'cfg', None), 'dim', 3)

    api.step(1e-3)
    hobj = getattr(provider, "_h", None)
    if hobj is None:
        raise RuntimeError("Softbody provider failed to initialize _h")

    n_cells = len(getattr(hobj, 'cells', []) or [])
    vtx_counts = _cells_vertex_counts(hobj)
    faces_concat, face_counts = _cells_faces_counts(hobj)
    vtx_offsets = np.zeros(n_cells + 1, dtype=np.int64)
    for i in range(n_cells):
        vtx_offsets[i + 1] = vtx_offsets[i] + int(vtx_counts[i])

    mvps = np.zeros((frames, 4, 4), dtype=np.float32)
    colors = np.zeros((frames, n_cells, 4), dtype=np.float32)
    draw_order = np.zeros((frames, n_cells), dtype=np.int32)
    vtx_concat = np.zeros((frames, int(vtx_offsets[-1]), 3), dtype=np.float32)

    center, radius = _compute_center_radius(hobj)
    eye = np.array([0.5, 0.5, 1.7], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam_dir = center - eye; cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
    cam_dist = float(np.linalg.norm(center - eye))
    center_s = center.copy(); dist_s = cam_dist
    aspect = w / max(1, h)

    t_sim = 0.0
    for f in range(frames):
        dt = step_cellsim(api, dt)
        t_sim += dt
        hobj = getattr(provider, "_h", hobj)
        if args.verbose or args.debug:
            _log_hierarchy_state(hobj, f, api=api, debug=args.debug)
        vtx_concat[f, :, :] = np.concatenate([np.asarray(c.X, dtype=np.float32) for c in hobj.cells], axis=0)

        pressures, masses, greens255 = _measure_pressure_mass(api, hobj)
        pN = _normalize(pressures)
        mN = _normalize(masses)
        BASE_R, BASE_B, BASE_A = 0.15, 0.25, 0.35
        PRESSURE_GAIN, MASS_GAIN = 0.75, 0.75
        G = (greens255.astype(np.float32) / 255.0)
        R = np.minimum(1.0, BASE_R + MASS_GAIN * mN.astype(np.float32))
        B = np.minimum(1.0, BASE_B + PRESSURE_GAIN * pN.astype(np.float32))
        colors[f, :, 0] = R
        colors[f, :, 1] = G
        colors[f, :, 2] = B
        colors[f, :, 3] = BASE_A

        new_center, radius = _compute_center_radius(hobj)
        desired_dist = max(0.2, radius / math.tan(math.radians(fovy * 0.5)) * 2.0)
        alpha = 0.15
        center_s = (1.0 - alpha) * center_s + alpha * new_center
        dist_s = (1.0 - alpha) * dist_s + alpha * desired_dist
        center = center_s; cam_dist = float(dist_s)
        eye = center - cam_dir * cam_dist
        P = _perspective(fovy, aspect, 0.05, max(10.0, cam_dist + 3.0 * radius))
        V = _look_at(eye, center, up)
        theta = rot_speed * t_sim if dim == 3 else 0.0
        T_neg = _translate(-center)
        R_y = _rotate_y(theta)
        T_pos = _translate(center)
        M = (T_pos @ R_y @ T_neg).astype(np.float32)
        MVP = (P @ V @ M).astype(np.float32)
        mvps[f] = MVP

        view_dir = (center - eye); view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)
        Xcat = vtx_concat[f]
        off = vtx_offsets.astype(np.int32)
        sums = np.add.reduceat(Xcat, off[:-1], axis=0)
        counts = (off[1:] - off[:-1]).astype(np.float32)[:, None]
        centroids = sums / np.maximum(counts, 1e-12)
        centroids_h = np.concatenate([centroids, np.ones((n_cells, 1), dtype=np.float32)], axis=1)
        ctr_rot = (M @ centroids_h.T).T[:, :3]
        depths = np.dot(ctr_rot - eye[None, :], view_dir)
        draw_order[f, :] = np.argsort(depths).astype(np.int32)

    play_mesh_stream(
        n_cells=n_cells,
        vtx_counts=vtx_counts,
        vtx_offsets=vtx_offsets,
        faces_concat=faces_concat,
        face_counts=face_counts,
        vtx_concat=vtx_concat,
        colors=colors,
        draw_order=draw_order,
        mvps=mvps,
        viewport_w=w,
        viewport_h=h,
        loop_mode=args.loop,
        fps=float(args.fps),
    )


def export_fluid_points_stream(args, gather_func, step_func, dim: int = 3):
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    w, h = int(args.gl_viewport_w), int(args.gl_viewport_h)
    fovy = float(args.gl_fovy)
    rot_speed = float(args.gl_rot_speed)

    pts_offsets = np.zeros(frames + 1, dtype=np.int64)
    pts_concat_list = []
    vec_concat_list = []
    mvps = np.zeros((frames, 4, 4), dtype=np.float32)

    pts0, vecs0 = gather_func()
    center, radius = _compute_center_radius_pts(pts0)
    eye = np.array([0.5, 0.5, 1.7], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam_dir = center - eye; cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
    cam_dist = float(np.linalg.norm(center - eye))
    center_s = center.copy(); dist_s = cam_dist
    aspect = w / max(1, h)

    t_sim = 0.0
    pts_offsets[0] = 0
    for f in range(frames):
        pts, vecs = gather_func()
        pts_concat_list.append(pts.astype(np.float32, copy=False))
        vec_concat_list.append(vecs.astype(np.float32, copy=False))
        pts_offsets[f + 1] = pts_offsets[f] + int(len(pts))

        new_center, radius = _compute_center_radius_pts(pts)
        desired_dist = max(0.2, radius / math.tan(math.radians(fovy * 0.5)) * 2.0)
        alpha = 0.15
        center_s = (1.0 - alpha) * center_s + alpha * new_center
        dist_s = (1.0 - alpha) * dist_s + alpha * desired_dist
        center = center_s; cam_dist = float(dist_s)
        eye = center - cam_dir * cam_dist

        P = _perspective(fovy, aspect, 0.05, max(10.0, cam_dist + 3.0 * radius))
        V = _look_at(eye, center, up)
        theta = rot_speed * t_sim if dim == 3 else 0.0
        T_neg = _translate(-center)
        R_y = _rotate_y(theta)
        T_pos = _translate(center)
        M = (T_pos @ R_y @ T_neg).astype(np.float32)
        mvps[f] = (P @ V @ M).astype(np.float32)

        step_func(dt)
        t_sim += dt

    pts_concat = (
        np.concatenate(pts_concat_list, axis=0)
        if pts_concat_list
        else np.zeros((0, 3), dtype=np.float32)
    )
    vec_concat = (
        np.concatenate(vec_concat_list, axis=0)
        if vec_concat_list
        else np.zeros((0, 3), dtype=np.float32)
    )
    meta = dict(stream_type='opengl_points_v1', frames=frames, viewport_w=w, viewport_h=h)
    np.savez_compressed(
        args.export_npz,
        **meta,
        pts_offsets=pts_offsets,
        pts_concat=pts_concat,
        vec_concat=vec_concat,
        mvps=mvps,
    )


def stream_fluid_points(
    args,
    gather_func,
    step_func,
    dim: int = 3,
    *,
    show_vectors: bool = False,
    color_metric: str = "magnitude",
    arrow_scale: float = 1.0,
    flow_anim_speed: float = 1.0,
):
    from .run_opengl_demo import play_points_stream
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    w, h = int(args.gl_viewport_w), int(args.gl_viewport_h)
    fovy = float(args.gl_fovy)
    rot_speed = float(args.gl_rot_speed)

    pts_offsets = np.zeros(frames + 1, dtype=np.int64)
    pts_concat_list = []
    vec_concat_list = [] if show_vectors else None
    scalar_concat_list = [] if show_vectors and color_metric != "none" else None
    mvps = np.zeros((frames, 4, 4), dtype=np.float32)

    pts0, _vecs0 = gather_func()
    center, radius = _compute_center_radius_pts(pts0)
    eye = np.array([0.5, 0.5, 1.7], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam_dir = center - eye; cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
    cam_dist = float(np.linalg.norm(center - eye))
    center_s = center.copy(); dist_s = cam_dist
    aspect = w / max(1, h)

    t_sim = 0.0
    pts_offsets[0] = 0
    for f in range(frames):
        pts, vecs = gather_func()
        pts_concat_list.append(pts.astype(np.float32, copy=False))
        if show_vectors and vecs is not None and vec_concat_list is not None:
            vec = vecs.astype(np.float32, copy=False)
            vec_concat_list.append(vec)
            if scalar_concat_list is not None and color_metric == "magnitude":
                scalar_concat_list.append(np.linalg.norm(vec, axis=1).astype(np.float32, copy=False))
        pts_offsets[f + 1] = pts_offsets[f] + int(len(pts))

        new_center, radius = _compute_center_radius_pts(pts)
        desired_dist = max(0.2, radius / math.tan(math.radians(fovy * 0.5)) * 2.0)
        alpha = 0.15
        center_s = (1.0 - alpha) * center_s + alpha * new_center
        dist_s = (1.0 - alpha) * dist_s + alpha * desired_dist
        center = center_s; cam_dist = float(dist_s)
        eye = center - cam_dir * cam_dist

        P = _perspective(fovy, aspect, 0.05, max(10.0, cam_dist + 3.0 * radius))
        V = _look_at(eye, center, up)
        theta = rot_speed * t_sim if dim == 3 else 0.0
        T_neg = _translate(-center)
        R_y = _rotate_y(theta)
        T_pos = _translate(center)
        M = (T_pos @ R_y @ T_neg).astype(np.float32)
        mvps[f] = (P @ V @ M).astype(np.float32)

        step_func(dt)
        t_sim += dt

    pts_concat = (
        np.concatenate(pts_concat_list, axis=0)
        if pts_concat_list
        else np.zeros((0, 3), dtype=np.float32)
    )
    vec_concat = (
        np.concatenate(vec_concat_list, axis=0)
        if vec_concat_list
        else None
    )
    scalar_concat = (
        np.concatenate(scalar_concat_list, axis=0)
        if scalar_concat_list
        else None
    )
    play_points_stream(
        pts_offsets,
        pts_concat,
        mvps,
        vec_concat=vec_concat,
        scalar_concat=scalar_concat,
        show_vectors=show_vectors,
        color_metric=color_metric,
        arrow_scale=arrow_scale,
        flow_anim_speed=flow_anim_speed,
        viewport_w=w,
        viewport_h=h,
        loop_mode=args.loop,
        fps=float(args.fps),
    )


def run_fluid_demo(args):
    if args.fluid == "discrete":
        from src.cells.bath.discrete_fluid import DiscreteFluid
        fluid = DiscreteFluid.demo_dam_break(n_x=8, n_y=12, n_z=8, h=0.05)
        step = fluid.step
    elif args.fluid == "voxel":
        from src.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams
        fluid = VoxelMACFluid(VoxelFluidParams(nx=8, ny=8, nz=8))
        step = fluid.step
    else:
        raise SystemExit("Unknown fluid engine")

    def gather():
        if hasattr(fluid, "export_positions_vectors"):
            return fluid.export_positions_vectors()
        if hasattr(fluid, "export_vector_field"):
            return fluid.export_vector_field()
        pts = fluid.export_vertices() if hasattr(fluid, "export_vertices") else np.zeros((0, 3), dtype=np.float32)
        return pts, None

    if getattr(args, "export_npz", "") and args.export_kind == "opengl-points":
        export_fluid_points_stream(args, gather, step)
        return

    if getattr(args, "stream", "") == "opengl-points":
        if args.fluid == "discrete":
            stream_fluid_points(
                args,
                fluid.export_positions_vectors,
                fluid.step,
                show_vectors=args.show_vectors,
                color_metric=args.color_metric,
                arrow_scale=args.arrow_scale,
                flow_anim_speed=args.flow_anim_speed,
            )
        else:
            stream_fluid_points(
                args,
                gather,
                step,
                show_vectors=args.show_vectors,
                color_metric=args.color_metric,
                arrow_scale=args.arrow_scale,
                flow_anim_speed=args.flow_anim_speed,
            )
        return

    dt = float(getattr(args, "dt", 1e-3))
    for _ in range(int(args.frames)):
        step(dt)

def main():
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)
    if getattr(args, "fluid", ""):
        run_fluid_demo(args)
        return

    api, provider = make_cellsim_backend(
        cell_vols=args.cell_vols,
        cell_imps=args.cell_imps,
        cell_elastic_k=args.cell_elastic_k,
        bath_na=args.bath_na,
        bath_cl=args.bath_cl,
        bath_pressure=args.bath_pressure,
        bath_volume_factor=args.bath_volume_factor,
        substeps=args.substeps,
        dt_provider=args.dt_provider,
    )
    # If export requested, produce stream file and exit
    if getattr(args, "export_npz", "") and getattr(args, "export_kind", ""):
        if args.export_kind == "ascii":
            export_ascii_stream(args, api, provider)
        elif args.export_kind == "opengl-points":
            export_opengl_points_stream(args, api, provider)
        elif args.export_kind == "opengl-mesh":
            export_opengl_mesh_stream(args, api, provider)
        else:
            raise SystemExit("Unknown export kind")
            logger.info("Wrote stream to %s", args.export_npz)
            return

    # Live in-process streaming: build arrays and hand to visualizer APIs
    if getattr(args, "stream", ""):
        kind = args.stream
        if kind == "ascii":
            stream_ascii(args, api, provider)
        elif kind == "opengl-points":
            stream_opengl_points(args, api, provider)
        elif kind == "opengl-mesh":
            stream_opengl_mesh(args, api, provider)
        else:
            raise SystemExit("Unknown --stream kind")
        return

    dt = args.dt
    # Use array state from engine if available
    engine = getattr(api, "engine", None)
    prev_vols = getattr(engine, "V", np.array([getattr(c, "V", 0.0) for c in api.cells], dtype=float)).astype(float, copy=False)
    for frame in range(int(args.frames)):
        dt = step_cellsim(api, dt)
        engine = getattr(api, "engine", engine)
        vols = getattr(engine, "V", np.array([getattr(c, "V", 0.0) for c in api.cells], dtype=float)).astype(float, copy=False)
        # dV (change in volume), kept as its own stat (not velocity)
        dV = vols - prev_vols
        # Compute COM velocities from softbody provider
        h = getattr(provider, "_h", None)
        v_out = None
        if h is not None and getattr(h, "cells", None):
            try:
                coms_vcoms = [_com_and_com_vel(c) for c in h.cells]
                _, vcoms = zip(*coms_vcoms) if coms_vcoms else ([], [])
                v_out = [tuple(float(x) for x in v) for v in vcoms]
            except Exception:
                v_out = None
        osm = getattr(engine, "osmotic_pressure", np.zeros_like(vols))
        # Lightweight textual output without converting arrays to Python lists unnecessarily
        if args.verbose or args.debug:
            _log_hierarchy_state(h, frame, api=api, debug=args.debug)
        msg = (
            f"vols {np.array2string(vols, precision=4)} dV {np.array2string(dV, precision=4)}"
        )
        if v_out is not None:
            msg += f" com_vel {v_out}"
        msg += f" osm {np.array2string(osm, precision=4)}"
        logger.info("frame %d: %s", frame, msg)
        prev_vols = vols


if __name__ == "__main__":
    main()
