import argparse
from typing import Sequence, Tuple, Dict, Any, Optional

import numpy as np
import logging
import os
import threading

from src.cells.cellsim.data.state import Cell, Bath
from src.cells.cellsim.api.saline import SalinePressureAPI
from src.cells.cellsim.mechanics.softbody0d import SoftbodyProviderCfg
from src.cells.bath.coupling import BathFluidCoupler
from src.cells.bath.dt_controller import STController, Targets
from src.common.sim_hooks import SimHooks

# Lightweight math helpers (duplicated to avoid importing OpenGL demo)
import math

# Generic layer packing for external renderers

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


def make_fluid_engine(kind: str, dim: int):
    """Construct a simple fluid engine for coupling or stand-alone runs."""
    kind = (kind or "").lower()
    if kind == "discrete":
        from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams

        params = FluidParams()
        # Place a tiny cross of particles for determinism
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.05],
            ],
            dtype=np.float64,
        )
        return DiscreteFluid(pts, None, None, None, params)
    if kind == "voxel":
        from src.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams

        nz = 4 if dim == 3 else 1
        params = VoxelFluidParams(nx=4, ny=4, nz=nz, dx=0.05)
        return VoxelMACFluid(params)
    if kind == "hybrid":
        from src.cells.bath.hybrid_fluid import HybridFluid, HybridParams

        nz = 4 if dim == 3 else 1
        shape = (4, 4, nz)
        return HybridFluid(shape, n_particles=8, params=HybridParams())
    raise ValueError(f"Unknown fluid kind: {kind}")


def _centers_and_vols(provider, api) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    h = getattr(provider, "_h", None)
    centers = []
    if h is not None:
        for c in getattr(h, "cells", []) or []:
            com, _ = _com_and_com_vel(c)
            centers.append(com)
    centers_arr = np.asarray(centers, dtype=float) if centers else None
    vols_arr = np.asarray([getattr(c, "V", 0.0) for c in getattr(api, "cells", [])], dtype=float)
    return centers_arr, vols_arr


def step_cellsim(api: SalinePressureAPI, dt: float, *, hooks=None) -> float:
    """Advance cellsim one step and finalize bath thermodynamics.

    Parameters
    ----------
    api:
        Cellsim API instance.
    dt:
        Timestep to advance.
    hooks:
        Optional :class:`~src.common.sim_hooks.SimHooks` passed to the engine.

    Returns
    -------
    float
        Suggested next ``dt`` from the adaptive controller.
    """

    dt = api.step(dt, hooks=hooks)
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
        "--sim-dim",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Spatial dimension for fluid and softbody sims",
    )
    parser.add_argument(
        "--fluid",
        choices=["", "discrete", "voxel", "hybrid"],
        default="",
        help="Run stand-alone fluid demo (discrete, voxel, or hybrid) instead of cellsim",
    )
    # Optional coupling: run cellsim with an external fluid engine
    parser.add_argument(
        "--couple-fluid",
        choices=["", "discrete", "voxel", "hybrid"],
        default="",
        help="Couple Bath/cellsim to a spatial fluid engine during cellsim run.",
    )
    parser.add_argument("--couple-radius", type=float, default=0.05,
                        help="Source influence radius for discrete fluid coupling (world units)")
    # Export/stream options
    parser.add_argument("--export-npz", type=str, default="",
                        help="If set, write a prerendered NPZ stream to this path.")
    parser.add_argument("--export-kind", choices=["ascii"], default="",
                        help="What kind of stream to export when --export-npz is set.")
    parser.add_argument("--stream", choices=["", "ascii"], default="",
                        help="If set, stream frames live to the chosen visualizer in-process (no files).")
    parser.add_argument("--stream-dir", type=str, default="",
                        help="If set, write uncompressed frames to this directory and stream via disk.")
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
    parser.add_argument("--debug-render", action="store_true",
                        help="Print gathered layers each frame instead of invoking OpenGL")
    parser.add_argument("--verbose", action="store_true", help="Log per-cell parameters each frame")
    parser.add_argument("--debug", action="store_true", help="Log full per-vertex and per-face data")
    return parser


def get_numpy_tag_names() -> list[str]:
    """Return the argparse dest names for the shared NumPy demo options.

    Useful for other modules (e.g., OpenGL demo) to extract only the
    NumPy-relevant kwargs from a larger argument namespace.
    """
    p = build_numpy_parser(add_help=False)
    # Exclude argparse internals like 'help'
    names = [a.dest for a in getattr(p, "_actions", []) if getattr(a, "dest", None)]
    names = [n for n in names if n not in ("help",)]
    return sorted(set(names))


def extract_numpy_kwargs(args_or_dict: argparse.Namespace | dict) -> dict:
    """Filter an args Namespace or dict down to NumPy demo kwargs only."""
    tags = set(get_numpy_tag_names())
    if isinstance(args_or_dict, dict):
        return {k: v for k, v in args_or_dict.items() if k in tags}
    return {k: getattr(args_or_dict, k) for k in tags if hasattr(args_or_dict, k)}


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

def stream_ascii(args, api, provider):
    # Stream frames live to the ASCII viewer
    from .run_ascii_demo import play_ascii_stream
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    nx, ny = int(args.ascii_nx), int(args.ascii_ny)
    counter = {"f": 0}

    def gather():
        h = getattr(provider, "_h", None)
        f = counter["f"]
        if args.verbose or args.debug:
            _log_hierarchy_state(h, f, api=api, debug=args.debug)
        if h is None:
            ch = np.zeros((ny, nx), dtype=np.uint8)
            col = np.zeros((ny, nx, 3), dtype=np.uint8)
        else:
            ch, col = _rasterize_ascii_numpy(
                h,
                api,
                nx,
                ny,
                render_mode=args.render_mode,
                face_stride=args.face_stride,
                draw_points=not args.no_points,
            )
        counter["f"] += 1
        return ch, col

    def step(dt_local):
        return step_cellsim(api, dt_local)

    play_ascii_stream(
        gather_func=gather,
        step_func=step,
        frames=frames,
        dt=dt,
        color_mode="auto",
        loop_mode=args.loop,
        fps=float(args.fps),
    )

def stream_ascii_to_dir(args, api, provider):
    from .run_ascii_demo import play_ascii_stream_from_dir
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    nx, ny = int(args.ascii_nx), int(args.ascii_ny)
    os.makedirs(args.stream_dir, exist_ok=True)
    done_path = os.path.join(args.stream_dir, "done")

    def writer():
        nonlocal dt
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
            np.save(os.path.join(args.stream_dir, f"chars_{f:06d}.npy"), ch)
            np.save(os.path.join(args.stream_dir, f"rgb_{f:06d}.npy"), col)
        open(done_path, "wb").close()

    t = threading.Thread(target=writer)
    t.start()
    play_ascii_stream_from_dir(args.stream_dir, loop_mode=args.loop, fps=float(args.fps))
    t.join()


def run_fluid_demo(args):
    engine = make_fluid_engine(args.fluid, args.sim_dim)
    dt = float(getattr(args, "dt", 1e-3))
    draw_hook = None
    if getattr(args, "debug_render", False):
        try:
            from src.opengl_render.renderer import DebugRenderer
            from src.opengl_render.api import make_draw_hook

            renderer = DebugRenderer()
            draw_hook = make_draw_hook(renderer, (0, 0))
        except Exception:  # pragma: no cover - debug aid
            def _fallback(layer_map):
                print(layer_map)

            draw_hook = _fallback
    for _ in range(int(args.frames)):
        engine.step(dt)
        if draw_hook is not None:
            layers = gather_layers(None, engine, for_opengl=False)
            draw_hook(layers)


def gather_layers(provider, fluid=None, *, rainbow: bool = False, for_opengl: bool = True):
    """Assemble visualization layers from the current simulators.

    ``provider`` may be ``None`` when only a fluid engine is present.  When
    ``for_opengl`` is ``True`` (default) the data is packed into
    :mod:`opengl_render` layer dataclasses.  When ``False`` a simpler mapping of
    plain ``numpy`` arrays is returned for debug printing or non-OpenGL
    consumers.
    """

    layers: Dict[str, Any] = {}
    h = getattr(provider, "_h", None) if provider is not None else None

    if for_opengl:
        from src.opengl_render.api import cellsim_layers, fluid_layers  # local import

        if h is not None:
            layers.update(cellsim_layers(h, rainbow=rainbow))
        if fluid is not None:
            try:
                layers.update(fluid_layers(fluid, rainbow=rainbow))
            except Exception:
                pass
        return layers

    # Generic arrays path
    if h is not None:
        positions: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        offset = 0
        for cell in getattr(h, "cells", []) or []:
            X = np.asarray(getattr(cell, "X", np.zeros((0, 3))), dtype=np.float32)
            F = np.asarray(getattr(cell, "F", np.zeros((0, 3), dtype=np.uint32)), dtype=np.uint32)
            if X.size and F.size:
                positions.append(X)
                faces.append(F + offset)
                offset += X.shape[0]
        if positions and faces:
            layers["membrane"] = {
                "positions": np.concatenate(positions),
                "faces": np.concatenate(faces),
            }
    if fluid is not None:
        pts = getattr(fluid, "p", None)
        if pts is not None:
            layers["fluid"] = {"points": np.asarray(pts, dtype=np.float32)}
    return layers


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
        dim=args.sim_dim,
    )
    fluid_engine = None
    coupler = None
    if getattr(args, "couple_fluid", ""):
        fluid_engine = make_fluid_engine(args.couple_fluid, args.sim_dim)
        coupler = BathFluidCoupler(api.bath, fluid_engine, kind=args.couple_fluid, radius=args.couple_radius)
        _, vols0 = _centers_and_vols(provider, api)
        if vols0 is not None:
            coupler.prime_volumes(vols0)

    if getattr(args, "export_npz", "") and getattr(args, "export_kind", "") == "ascii":
        export_ascii_stream(args, api, provider)
        logger.info("Wrote stream to %s", args.export_npz)
        return
    if getattr(args, "stream", "") == "ascii":
        if getattr(args, "stream_dir", ""):
            stream_ascii_to_dir(args, api, provider)
        else:
            stream_ascii(args, api, provider)
        return
    draw_hook = None
    if args.debug_render:
        from src.opengl_render.renderer import DebugRenderer
        from src.opengl_render.api import make_draw_hook

        renderer = DebugRenderer()
        draw_hook = make_draw_hook(renderer, (0, 0))

    dt = float(getattr(args, "dt", 1e-3))
    hooks = SimHooks()
    for _ in range(int(args.frames)):
        if coupler is not None:
            centers, vols = _centers_and_vols(provider, api)
            coupler.exchange(dt=dt, centers=centers, vols=vols, hooks=hooks)
        dt = step_cellsim(api, dt, hooks=hooks)
        if draw_hook is not None:
            layers = gather_layers(provider, fluid_engine, rainbow=False, for_opengl=False)
            draw_hook(layers)

if __name__ == "__main__":
    main()
