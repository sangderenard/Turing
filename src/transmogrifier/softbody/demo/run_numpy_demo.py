import argparse
from typing import Sequence, Tuple, Dict, Any, Optional

import numpy as np

from src.transmogrifier.cells.cellsim.data.state import Cell, Bath
from src.transmogrifier.cells.cellsim.api.saline import SalinePressureAPI
from src.transmogrifier.cells.cellsim.mechanics.softbody0d import SoftbodyProviderCfg

# Lightweight math helpers (duplicated to avoid importing OpenGL demo)
import math

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
        SoftbodyProviderCfg(substeps=substeps, dt_provider=dt_provider)
    )
    return api, provider


def step_cellsim(api: SalinePressureAPI, dt: float) -> float:
    """Advance cellsim one step; returns suggested next dt."""
    return api.step(dt)


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
    parser.add_argument("--cell-vols", type=float, nargs="+", default=[1.6, 1.2, 0.9])
    parser.add_argument("--cell-imps", type=float, nargs="+", default=[100.0, 130.0, 160.0])
    parser.add_argument("--cell-elastic-k", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    parser.add_argument("--bath-volume-factor", type=float, default=5.0)
    parser.add_argument("--bath-na", type=float, default=1000.0)
    parser.add_argument("--bath-cl", type=float, default=1000.0)
    parser.add_argument("--bath-pressure", type=float, default=1e4)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument(
        "--dt-provider", type=float, default=0.01,
        help="internal softbody timestep; scale to speed up motion",
    )
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument(
        "--dt", type=float, default=1e-10,
        help="base integrator step; increase to amplify drift",
    )
    # Export options
    parser.add_argument("--export-npz", type=str, default="",
                        help="If set, write a prerendered NPZ stream to this path.")
    parser.add_argument("--export-kind", choices=["ascii", "opengl-points", "opengl-mesh"], default="",
                        help="What kind of stream to export when --export-npz is set.")
    parser.add_argument("--ascii-nx", type=int, default=120, help="ASCII export grid width")
    parser.add_argument("--ascii-ny", type=int, default=36, help="ASCII export grid height")
    parser.add_argument("--render-mode", choices=["edges", "fill"], default="edges",
                        help="ASCII export render mode (edges/fill)")
    parser.add_argument("--face-stride", type=int, default=8,
                        help="ASCII export: draw every Nth face (performance)")
    parser.add_argument("--no-points", action="store_true",
                        help="ASCII export: disable drawing vertex points")
    parser.add_argument("--gl-fovy", type=float, default=45.0, help="OpenGL stream: vertical FOV degrees")
    parser.add_argument("--gl-rot-speed", type=float, default=0.25, help="OpenGL stream: Y-rotation speed rad/sec (used as frame*dt)")
    parser.add_argument("--gl-viewport-w", type=int, default=1100, help="OpenGL stream: viewport width")
    parser.add_argument("--gl-viewport-h", type=int, default=800, help="OpenGL stream: viewport height")
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

def _compute_center_radius(h) -> Tuple[np.ndarray, float]:
    allX = np.concatenate([c.X for c in h.cells], axis=0).astype(np.float32)
    bmin, bmax = allX.min(0), allX.max(0)
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

def export_ascii_stream(args, api, provider):
    # Import here to avoid circular import at module load time
    from .run_ascii_demo import world_to_grid
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    nx, ny = int(args.ascii_nx), int(args.ascii_ny)
    chars = np.zeros((frames, ny, nx), dtype=np.uint8)
    rgb = np.zeros((frames, ny, nx, 3), dtype=np.uint8)

    for f in range(frames):
        dt = step_cellsim(api, dt)
        h = getattr(provider, "_h", None)
        if h is None:
            continue
        grid = world_to_grid(
            h,
            api=api,
            nx=nx,
            ny=ny,
            render_mode=args.render_mode,
            face_stride=args.face_stride,
            draw_points=not args.no_points,
        )
        # Convert grid to arrays
        for iy, row in enumerate(grid):
            for ix, cell in enumerate(row):
                if isinstance(cell, str):
                    ch = cell
                    col = (255, 255, 255)
                else:
                    ch, col = cell
                chars[f, iy, ix] = ord(ch[0]) if ch else 32
                r, g, b = col
                rgb[f, iy, ix, 0] = int(r)
                rgb[f, iy, ix, 1] = int(g)
                rgb[f, iy, ix, 2] = int(b)

    meta = dict(stream_type='ascii_v1', nx=nx, ny=ny, frames=frames)
    np.savez_compressed(args.export_npz, **meta, chars=chars, rgb=rgb)

def export_opengl_points_stream(args, api, provider):
    dt = float(getattr(args, "dt", 1e-3))
    frames = int(args.frames)
    w, h = int(args.gl_viewport_w), int(args.gl_viewport_h)
    fovy = float(args.gl_fovy)
    rot_speed = float(args.gl_rot_speed)

    pts_offsets = np.zeros(frames + 1, dtype=np.int64)
    pts_concat_list = []
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
        theta = rot_speed * t_sim  # tie rotation to simulated time
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

        # fill vertex positions into concat buffer
        off = 0
        for i, c in enumerate(hobj.cells):
            X = np.asarray(c.X, dtype=np.float32)
            n = len(X)
            vtx_concat[f, off:off+n, :] = X
            off += n

        # colors from pressures/masses
        pressures, masses, greens255 = _measure_pressure_mass(api, hobj)
        pN = _normalize(pressures)
        mN = _normalize(masses)
        BASE_R, BASE_B, BASE_A = 0.15, 0.25, 0.35
        PRESSURE_GAIN, MASS_GAIN = 0.75, 0.75
        for i in range(n_cells):
            G = float(greens255[i]) / 255.0
            R = min(1.0, BASE_R + MASS_GAIN * float(mN[i]))
            B = min(1.0, BASE_B + PRESSURE_GAIN * float(pN[i]))
            colors[f, i, :] = np.array([R, G, B, BASE_A], dtype=np.float32)

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
        theta = rot_speed * t_sim
        T_neg = _translate(-center)
        R_y = _rotate_y(theta)
        T_pos = _translate(center)
        M = (T_pos @ R_y @ T_neg).astype(np.float32)
        MVP = (P @ V @ M).astype(np.float32)
        mvps[f] = MVP

        # depth sort order using centroid depth after model rotation
        view_dir = (center - eye); view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)
        depths = []
        for i, c in enumerate(hobj.cells):
            ctr = np.mean(c.X, axis=0).astype(np.float32)
            ctr_h = np.array([ctr[0], ctr[1], ctr[2], 1.0], dtype=np.float32)
            ctr_rot = (M @ ctr_h)[:3]
            d = float(np.dot(ctr_rot - eye, view_dir))
            depths.append((d, i))
        depths.sort(key=lambda x: x[0])
        draw_order[f, :] = np.array([i for _, i in depths], dtype=np.int32)

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

def main():
    args = parse_args()
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
        print(f"Wrote stream to {args.export_npz}")
        return

    dt = args.dt
    prev_vols = np.array([float(c.V) for c in api.cells], dtype=float)
    for frame in range(int(args.frames)):
        dt = step_cellsim(api, dt)
        vols = np.array([float(c.V) for c in api.cells], dtype=float)
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
        osm = np.array([getattr(c, "osmotic_pressure", 0.0) for c in api.cells], dtype=float)
        if v_out is None:
            print(f"frame {frame}: vols {vols.tolist()} dV {dV.tolist()} osm {osm.tolist()}")
        else:
            print(f"frame {frame}: vols {vols.tolist()} dV {dV.tolist()} com_vel {v_out} osm {osm.tolist()}")
        prev_vols = vols


if __name__ == "__main__":
    main()
