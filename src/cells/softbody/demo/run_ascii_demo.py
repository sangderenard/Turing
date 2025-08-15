import os, sys, argparse
import time, random, math
import numpy as np

# Keep local XPBD types for rendering helpers only
from ..engine.hierarchy import Hierarchy

# Reuse the numpy-based backend so all demos share identical math
from .run_numpy_demo import (
    make_cellsim_backend,
    step_cellsim,
    build_numpy_parser,
    get_numpy_tag_names,
    extract_numpy_kwargs,
)

# ---- color helpers ---------------------------------------------------------
try:
    import colorama  # makes Windows consoles understand ANSI
    colorama.just_fix_windows_console()
    _COLORAMA_OK = True
except Exception:
    _COLORAMA_OK = False

def _is_tty():
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

def _supports_truecolor():
    ct = os.environ.get("COLORTERM", "").lower()
    return ("truecolor" in ct) or ("24bit" in ct)

def _rgb256_from_24(r, g, b):
    def to_6(x): return max(0, min(5, int((x/255)*5 + 0.5)))
    r6, g6, b6 = to_6(r), to_6(g), to_6(b)
    idx = 16 + 36*r6 + 6*g6 + b6
    return f"\x1b[38;5;{idx}m"

RESET = "\x1b[0m"

def colorize(ch, rgb, mode="auto"):
    """
    mode: 'auto' | 'always' | 'never'
    - auto: only color if stdout is a TTY; otherwise plain text
    - always: force color (truecolor if available, else 256-color)
    - never: no ANSI at all
    """
    r, g, b = rgb
    if mode == "never":
        return ch
    if mode == "auto" and not _is_tty():
        return ch
    if _supports_truecolor():
        return f"\x1b[38;2;{r};{g};{b}m{ch}{RESET}"
    else:
        return f"{_rgb256_from_24(r,g,b)}{ch}{RESET}"


def world_to_grid(
    h: Hierarchy,
    api=None,
    nx=120,
    ny=36,
    render_mode: str = "edges",
    face_stride: int = 8,
    draw_points: bool = True,
    edge_ch: str = "#",
    fill_ch: str = "+",
):
    """Rasterize the softbody mesh onto an ASCII grid.

    ``edge_ch`` and ``fill_ch`` allow callers to specify separate characters for
    triangle edges versus interiors when ``render_mode`` is ``fill``.
    """
    grid = [[('.', (110, 110, 130)) for _ in range(nx)] for _ in range(ny)]
    cells = h.cells

    # Per-cell color channels (R: mass/solute proxy, G: identity, B: pressure)
    pressures = [c.contact_pressure_estimate() for c in cells]
    pmax = max(1e-8, max(pressures) if pressures else 0.0)
    masses = []
    if api is not None and getattr(api, "cells", None):
        for i in range(len(cells)):
            nd = getattr(api.cells[i], "n", None)
            masses.append(sum(nd.values()) if isinstance(nd, dict) else 0.0)
    else:
        masses = [getattr(c, "osmotic_pressure", 0.0) for c in cells]
    mmax = max(1e-8, max(masses) if masses else 0.0)

    # Helpers --------------------------------------------------------------
    def clamp_ixy(x: float, y: float):
        ix = max(0, min(nx - 1, int(x * nx)))
        iy = max(0, min(ny - 1, int(y * ny)))
        return ix, iy

    def put(ix: int, iy: int, ci: int, ch: str):
        cell = cells[ci]
        default_g = 64 + (ci % 3) * 80  # [64,144,224]
        G = getattr(
            cell,
            "_identity_green",
            getattr(getattr(api, "cells", [None] * len(cells))[ci], "_identity_green", default_g),
        )
        B = int(255 * (pressures[ci] / pmax)) if pmax > 0 else 0
        R = int(255 * (masses[ci] / mmax)) if mmax > 0 else 0
        grid[iy][ix] = (ch, (R, int(G), B))

    def draw_line(ix0: int, iy0: int, ix1: int, iy1: int, ci: int, ch: str = edge_ch):
        # Bresenham's line algorithm on grid indices
        dx = abs(ix1 - ix0)
        dy = -abs(iy1 - iy0)
        sx = 1 if ix0 < ix1 else -1
        sy = 1 if iy0 < iy1 else -1
        err = dx + dy
        x, y = ix0, iy0
        while True:
            if 0 <= x < nx and 0 <= y < ny:
                put(x, y, ci, ch)
            if x == ix1 and y == iy1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    # Rasterize meshes -----------------------------------------------------
    stride = max(1, int(face_stride))
    for ci, c in enumerate(cells):
        X = np.asarray(c.X)
        F = np.asarray(c.faces, dtype=np.int32)
        if X.ndim != 2 or X.shape[1] < 2 or F.size == 0:
            continue

        # Draw vertices as points
        if draw_points:
            # Subsample vertices to avoid overdraw: step by k based on grid size
            k = max(1, int(len(X) / (nx * 0.75)))
            for vi in range(0, len(X), k):
                x, y = float(X[vi, 0]), float(X[vi, 1])
                ix, iy = clamp_ixy(x, y)
                put(ix, iy, ci, '.')

        # Draw triangle edges or segments (subsample faces for speed)
        if render_mode in ("edges", "fill"):
            for fi in range(0, len(F), stride):
                row = F[fi]
                # Numpy row can be np.ndarray; get length robustly
                rlen = int(row.shape[0]) if hasattr(row, 'shape') else len(row)
                if rlen >= 3:
                    a, b, d = int(row[0]), int(row[1]), int(row[2])
                    # Triangle is (a, b, d) in our face layout
                    ax, ay = float(X[a, 0]), float(X[a, 1])
                    bx, by = float(X[b, 0]), float(X[b, 1])
                    dx_, dy_ = float(X[d, 0]), float(X[d, 1])
                    iax, iay = clamp_ixy(ax, ay)
                    ibx, iby = clamp_ixy(bx, by)
                    idx, idy = clamp_ixy(dx_, dy_)
                    draw_line(iax, iay, ibx, iby, ci, edge_ch)
                    draw_line(ibx, iby, idx, idy, ci, edge_ch)
                    draw_line(idx, idy, iax, iay, ci, edge_ch)

                    if render_mode == "fill":
                        # Barycentric rasterization in bounding box
                        minx = max(0, min(iax, ibx, idx))
                        maxx = min(nx - 1, max(iax, ibx, idx))
                        miny = max(0, min(iay, iby, idy))
                        maxy = min(ny - 1, max(iay, iby, idy))

                        # Precompute area for barycentric weights
                        denom = (iby - idy) * (iax - idx) + (idx - ibx) * (iay - idy)
                        if abs(denom) < 1e-8:
                            continue  # degenerate
                        for yy in range(miny, maxy + 1):
                            for xx in range(minx, maxx + 1):
                                w1 = ((iby - idy) * (xx - idx) + (idx - ibx) * (yy - idy)) / denom
                                w2 = ((idy - iay) * (xx - idx) + (iax - idx) * (yy - idy)) / denom
                                w3 = 1.0 - w1 - w2
                                if w1 >= 0 and w2 >= 0 and w3 >= 0:
                                    put(xx, yy, ci, fill_ch)
                elif rlen == 2:
                    # Segment face (common in 1D sims): just draw the edge
                    a, b = int(row[0]), int(row[1])
                    ax, ay = float(X[a, 0]), float(X[a, 1])
                    bx, by = float(X[b, 0]), float(X[b, 1])
                    iax, iay = clamp_ixy(ax, ay)
                    ibx, iby = clamp_ixy(bx, by)
                    draw_line(iax, iay, ibx, iby, ci, edge_ch)
                else:
                    # Unsupported face size; skip
                    continue

        # Draw organelles as small discs
        for o in getattr(c, 'organelles', []) or []:
            cx, cy = float(o.pos[0]), float(o.pos[1])
            # Radius in world units -> pixels
            pr = max(1, int(o.radius * max(nx, ny)))
            icx, icy = clamp_ixy(cx, cy)
            rr = pr * pr
            for dy in range(-pr, pr + 1):
                py = icy + dy
                if py < 0 or py >= ny:
                    continue
                for dx in range(-pr, pr + 1):
                    px = icx + dx
                    if px < 0 or px >= nx:
                        continue
                    if dx * dx + dy * dy <= rr:
                        put(px, py, ci, 'o')

    return grid

def print_grid(grid, color_mode="auto", clear_each_line=False):
    """Print the grid and return the number of lines printed (no clearing)."""
    count = 0
    for row in grid:
        line = ''.join([colorize(ch, (r, g, b), mode=color_mode) for ch, (r, g, b) in row])
        # Ensure we start at column 1 of the current line without clearing
        print("\x1b[G" + line)
        count += 1
    return count


def world_to_line(h: Hierarchy, api=None, nx: int = 120, line_ch: str = "="):
    """Rasterize the softbody mesh onto a single ASCII line.

    Each cell covers the horizontal span of its projected vertices.  The
    returned grid has shape ``[(ny=1, nx)]`` to remain compatible with
    ``print_grid``.
    """
    grid = [[('.', (110, 110, 130)) for _ in range(nx)]]
    cells = h.cells

    pressures = [c.contact_pressure_estimate() for c in cells]
    pmax = max(1e-8, max(pressures) if pressures else 0.0)
    masses = []
    if api is not None and getattr(api, "cells", None):
        for i in range(len(cells)):
            nd = getattr(api.cells[i], "n", None)
            masses.append(sum(nd.values()) if isinstance(nd, dict) else 0.0)
    else:
        masses = [getattr(c, "osmotic_pressure", 0.0) for c in cells]
    mmax = max(1e-8, max(masses) if masses else 0.0)

    for ci, c in enumerate(cells):
        X = np.asarray(c.X)
        if X.ndim != 2 or X.shape[1] < 2:
            continue
        xs = X[:, 0]
        x0 = max(0, min(nx - 1, int(xs.min() * nx)))
        x1 = max(0, min(nx - 1, int(xs.max() * nx)))
        default_g = 64 + (ci % 3) * 80
        G = getattr(
            c,
            "_identity_green",
            getattr(getattr(api, "cells", [None] * len(cells))[ci], "_identity_green", default_g),
        )
        B = int(255 * (pressures[ci] / pmax)) if pmax > 0 else 0
        R = int(255 * (masses[ci] / mmax)) if mmax > 0 else 0
        for ix in range(min(x0, x1), max(x0, x1) + 1):
            grid[0][ix] = (line_ch, (R, int(G), B))
    return grid

def main():
    # Build CLI that inherits all numpy demo params, plus ASCII-only flags
    parent = build_numpy_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[parent], conflict_handler='resolve')
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto",
                        help="ANSI color output mode (default: auto)")
    parser.add_argument("--frames", type=int, default=8000,
                        help="Number of frames to render (default: 8000)")
    parser.add_argument("--render-mode", choices=["edges", "fill"], default="edges",
                        help="ASCII render mode: project mesh edges (default) or fill (edges only for now)")
    parser.add_argument("--face-stride", type=int, default=8,
                        help="Draw every Nth face to control cost (default: 8)")
    parser.add_argument("--no-points", action="store_true",
                        help="Disable drawing vertex points to lighten output")
    parser.add_argument("--no-self-contacts", action="store_true",
                        help="Disable intra-mesh self-contact broad-phase to reduce RAM/CPU")
    parser.add_argument("--stream-npz", type=str, default="",
                        help="If provided, read a prerendered ASCII NPZ stream and play it.")
    parser.add_argument("--dim", choices=["2d", "1d"], default="2d",
                        help="Render as full 2D grid or a single 1D row")
    parser.add_argument("--edge-char", type=str, default="#",
                        help="Character for mesh edges")
    parser.add_argument("--fill-char", type=str, default="+",
                        help="Character for triangle interiors when render-mode=fill")
    parser.add_argument("--line-char", type=str, default="=",
                        help="Character for 1D cell spans")
    args = parser.parse_args()
    color_mode = args.color

    # If streaming from NPZ, bypass simulation entirely
    if getattr(args, "stream_npz", ""):
        data = np.load(args.stream_npz)
        stype = data['stream_type'].item() if ('stream_type' in getattr(data, 'files', [])) else None
        if stype != "ascii_v1":
            print(f"Unsupported stream type: {stype}")
            return
        chars = data["chars"]  # (F, ny, nx) uint8
        rgb = data["rgb"]      # (F, ny, nx, 3) uint8
        frames = chars.shape[0]
        prev_lines = 0
        for frame in range(frames):
            if _is_tty() and prev_lines:
                sys.stdout.write(f"\x1b[{prev_lines}F\x1b[G")
            sys.stdout.write(f"Frame {frame}\n")
            ny, nx = chars.shape[1], chars.shape[2]
            for iy in range(ny):
                parts = []
                for ix in range(nx):
                    ch = chr(int(chars[frame, iy, ix]))
                    col = tuple(int(x) for x in rgb[frame, iy, ix])
                    parts.append(colorize(ch, col, mode=color_mode))
                sys.stdout.write(''.join(parts) + "\n")
            sys.stdout.flush()
            prev_lines = 1 + ny
        return

    # Use the exact same backend parameters as the numpy demo, via args
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
    # Optional: prime mechanics and disable self-contacts before main loop
    if getattr(args, "no_self_contacts", False):
        try:
            # ensure hierarchy exists
            api.step(1e-6)
            h = getattr(provider, "_h", None)
            if h is not None and hasattr(h, "params"):
                setattr(h.params, "enable_self_contacts", False)
        except Exception:
            pass
    # Assign clear identity greens for cells (rendering only)
    try:
        levels = [64, 144, 208]
        for i, c in enumerate(getattr(api, "cells", []) or []):
            setattr(c, "_identity_green", levels[i % len(levels)])
    except Exception:
        pass
    dt = float(getattr(args, "dt", 1e-3))
    prev_lines = 0

    for frame in range(int(args.frames)):
        dt = step_cellsim(api, dt)
        h = getattr(provider, "_h", None)
        if h is None:
            continue

        if args.dim == "1d":
            grid = world_to_line(h, api=api, nx=120, line_ch=args.line_char)
        else:
            grid = world_to_grid(
                h,
                api=api,
                nx=120,
                ny=36,
                render_mode=args.render_mode,
                face_stride=args.face_stride,
                draw_points=not args.no_points,
                edge_ch=args.edge_char,
                fill_ch=args.fill_char,
            )

        if _is_tty():
            if prev_lines:
                # move to start of the previously printed block (NO newline)
                sys.stdout.write(f"\x1b[{prev_lines}F\x1b[G")

            # header
            sys.stdout.write(f"Frame {frame}\n")

            # write every row, colorized per cell (no clearing anywhere)
            if isinstance(grid, (list, tuple)):
                for row in grid:
                    if isinstance(row, str):
                        sys.stdout.write(row + "\n")
                    else:
                        # row is iterable of (ch,(r,g,b)) tuples
                        parts = []
                        for x in row:
                            if isinstance(x, str):
                                parts.append(x)
                            else:
                                ch, rgb = x
                                parts.append(colorize(ch, rgb, mode=color_mode))
                        sys.stdout.write(''.join(parts) + "\n")
            else:
                # tolerate a single string grid
                sys.stdout.write(str(grid) + "\n")

            sys.stdout.flush()
            prev_lines = 1 + (len(grid) if hasattr(grid, "__len__") else 1)
        else:
            print(f"Frame {frame}")
            if isinstance(grid, (list, tuple)):
                for row in grid:
                    print(row if isinstance(row, str) else ''.join(
                        (x if isinstance(x, str) else x[0]) for x in row))
                prev_lines = 1 + len(grid)
            else:
                print(grid)
                prev_lines = 2

if __name__ == '__main__':
    main()

# ---- NumPy tag bridge -------------------------------------------------------
def numpy_tag_names() -> list[str]:
    """Expose the shared NumPy CLI tag names for external callers."""
    return get_numpy_tag_names()


def numpy_kwargs_from_args(args: dict | object) -> dict:
    """Extract only NumPy-relevant kwargs from a larger args dict/namespace."""
    if isinstance(args, dict):
        return extract_numpy_kwargs(args)
    return {k: getattr(args, k) for k in numpy_tag_names() if hasattr(args, k)}

# ---- In-process ASCII streaming API ---------------------------------------
def play_ascii_stream(chars: np.ndarray,
                      rgb: np.ndarray,
                      *,
                      color_mode: str = "auto",
                      loop_mode: str = "none",
                      fps: float = 30.0):
    """Render an ASCII stream from NumPy arrays without file I/O.

    chars: (F, ny, nx) uint8 of ASCII codes.
    rgb:   (F, ny, nx, 3) uint8 of per-cell colors.
    loop_mode: 'none' | 'loop' | 'bounce'
    """
    import time
    F = int(chars.shape[0])
    frame = 0
    direction = 1
    prev_lines = 0
    dt_target = 1.0 / max(1e-6, float(fps))
    while True:
        fidx = int(frame)
        if fidx >= F:
            if loop_mode == "loop":
                fidx = 0; frame = 0
            elif loop_mode == "bounce":
                direction = -1; frame = F-1; fidx = frame
            else:
                break
        elif fidx < 0 and loop_mode == "bounce":
            direction = 1; frame = 0; fidx = 0

        if _is_tty() and prev_lines:
            sys.stdout.write(f"\x1b[{prev_lines}F\x1b[G")
        sys.stdout.write(f"Frame {fidx}\n")
        ny, nx = chars.shape[1], chars.shape[2]
        for iy in range(ny):
            parts = []
            ch_row = chars[fidx, iy]
            rgb_row = rgb[fidx, iy]
            # vectorized join still needs per-cell colorization, but avoid Python tuple packing
            for ix in range(nx):
                ch = chr(int(ch_row[ix]))
                r = int(rgb_row[ix, 0]); g = int(rgb_row[ix, 1]); b = int(rgb_row[ix, 2])
                parts.append(colorize(ch, (r, g, b), mode=color_mode))
            sys.stdout.write(''.join(parts) + "\n")
        sys.stdout.flush()
        prev_lines = 1 + ny

        t0 = time.perf_counter()
        time.sleep(max(0.0, dt_target - (time.perf_counter() - t0)))
        frame += direction


def play_ascii_stream_from_dir(dir_path: str,
                               *,
                               color_mode: str = "auto",
                               loop_mode: str = "none",
                               fps: float = 30.0) -> None:
    """Render an ASCII stream by loading frames from ``dir_path``.

    Frames are expected as ``chars_XXXXXX.npy`` and ``rgb_XXXXXX.npy`` where
    ``XXXXXX`` is a zero-padded frame index.  A file named ``done`` in the
    directory signals that no further frames will be written.
    """
    import os
    import time

    frame = 0
    direction = 1
    prev_lines = 0
    dt_target = 1.0 / max(1e-6, float(fps))

    while True:
        ch_path = os.path.join(dir_path, f"chars_{frame:06d}.npy")
        rgb_path = os.path.join(dir_path, f"rgb_{frame:06d}.npy")

        while not (os.path.exists(ch_path) and os.path.exists(rgb_path)):
            if os.path.exists(os.path.join(dir_path, "done")):
                break
            time.sleep(0.05)

        if not (os.path.exists(ch_path) and os.path.exists(rgb_path)):
            if loop_mode == "loop":
                frame = 0
                direction = 1
                continue
            if loop_mode == "bounce":
                direction = -direction
                frame += direction
                if frame < 0:
                    frame = 0
                    direction = 1
                continue
            break

        ch = np.load(ch_path)
        rgb = np.load(rgb_path)
        if _is_tty() and prev_lines:
            sys.stdout.write(f"\x1b[{prev_lines}F\x1b[G")
        sys.stdout.write(f"Frame {frame}\n")
        ny, nx = ch.shape[0], ch.shape[1]
        for iy in range(ny):
            parts = []
            ch_row = ch[iy]
            rgb_row = rgb[iy]
            for ix in range(nx):
                c = chr(int(ch_row[ix]))
                r = int(rgb_row[ix, 0]); g = int(rgb_row[ix, 1]); b = int(rgb_row[ix, 2])
                parts.append(colorize(c, (r, g, b), mode=color_mode))
            sys.stdout.write(''.join(parts) + "\n")
        sys.stdout.flush()
        prev_lines = 1 + ny

        t0 = time.perf_counter()
        time.sleep(max(0.0, dt_target - (time.perf_counter() - t0)))
        frame += direction

