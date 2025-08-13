import argparse
from typing import Sequence

import numpy as np

from src.transmogrifier.cells.cellsim.data.state import Cell, Bath
from src.transmogrifier.cells.cellsim.api.saline import SalinePressureAPI
from src.transmogrifier.cells.cellsim.mechanics.softbody0d import SoftbodyProviderCfg


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
    return parser


def parse_args():
    return build_numpy_parser(add_help=True).parse_args()


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
