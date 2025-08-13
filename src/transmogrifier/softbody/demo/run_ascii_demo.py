import os, sys, argparse
import time, random, math
import numpy as np

# Keep local XPBD types for rendering helpers only
from ..engine.hierarchy import Hierarchy

# Reuse the numpy-based backend so all demos share identical math
from .run_numpy_demo import make_cellsim_backend, step_cellsim

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


def world_to_grid(h: Hierarchy, nx=120, ny=36):
    grid = [[('.', (110,110,130)) for _ in range(nx)] for _ in range(ny)]
    cells = h.cells
    centers = [np.mean(c.X, axis=0) for c in cells]
    vols = [abs(c.enclosed_volume()) for c in cells]
    radii = [((3.0*V)/(4.0*math.pi))**(1.0/3.0) for V in vols]
    pressures = [c.contact_pressure_estimate() for c in cells]
    pmax = max(1e-8, max(pressures))
    concs = [c.osmotic_pressure for c in cells]
    cmax = max(1e-8, max(concs))

    for iy in range(ny):
        y = (iy+0.5)/ny
        for ix in range(nx):
            x = (ix+0.5)/nx
            occup = None
            # organelles
            for j,c in enumerate(cells):
                for o in c.organelles:
                    dx = x - float(o.pos[0]); dy = y - float(o.pos[1])
                    if dx*dx + dy*dy <= (o.radius*o.radius):
                        occup=('o', j); break
                if occup: break
            if not occup:
                # cell body proxy
                for j,(ctr,r) in enumerate(zip(centers, radii)):
                    dx = x - float(ctr[0]); dy = y - float(ctr[1])
                    if dx*dx + dy*dy <= r*r:
                        occup=('#', j); break
            if occup:
                ch, ci = occup
                cell = cells[ci]
                G = getattr(cell, "_identity_green", 20 + ci * 20)
                B = int(255 * (pressures[ci]/pmax)) if pmax>0 else 0
                R = int(255 * (concs[ci]/cmax)) if cmax>0 else 0
                grid[iy][ix] = (ch, (R,G,B))
    return grid

def print_grid(grid, color_mode="auto"):
    lines = []
    for row in grid:
        line = ''.join([colorize(ch, (r, g, b), mode=color_mode) for ch,(r,g,b) in row])
        lines.append(line)
    print('\n'.join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto",
            help="ANSI color output mode (default: auto)")
    parser.add_argument("--frames", type=int, default=80)
    args = parser.parse_args()
    color_mode = args.color

    api, provider = make_cellsim_backend(
        cell_vols=[1.6, 1.2, 0.9],
        cell_imps=[100.0 + 30 * i for i in range(3)],
        cell_elastic_k=[0.6 + 0.1 * i for i in range(3)],
        bath_na=1000.0,
        bath_cl=1000.0,
        bath_pressure=1e4,
        bath_volume_factor=5.0,
        substeps=2,
        dt_provider=0.01,
    )
    dt = 1e-3

    for frame in range(int(args.frames)):
        dt = step_cellsim(api, dt)
        h = getattr(provider, "_h", None)
        if h is None:
            continue
        grid = world_to_grid(h, nx=120, ny=36)
        if _is_tty():
            print("\x1b[2J\x1b[H", end='')
        print(f"Frame {frame}")
        print_grid(grid, color_mode=color_mode)
        try:
            time.sleep(0.03)
        except Exception:
            pass

if __name__ == '__main__':
    main()
