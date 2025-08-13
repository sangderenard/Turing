import argparse
from typing import Sequence

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run softbody cellsim with numpy-only backend")
    parser.add_argument("--cell-vols", type=float, nargs="+", default=[1.6, 1.2, 0.9])
    parser.add_argument("--cell-imps", type=float, nargs="+", default=[100.0, 130.0, 160.0])
    parser.add_argument("--cell-elastic-k", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    parser.add_argument("--bath-volume-factor", type=float, default=5.0)
    parser.add_argument("--bath-na", type=float, default=1000.0)
    parser.add_argument("--bath-cl", type=float, default=1000.0)
    parser.add_argument("--bath-pressure", type=float, default=1e4)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--dt-provider", type=float, default=0.01)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--dt", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    api, _provider = make_cellsim_backend(
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
    for frame in range(int(args.frames)):
        dt = step_cellsim(api, dt)
        vols = [float(c.V) for c in api.cells]
        osm = [getattr(c, "osmotic_pressure", 0.0) for c in api.cells]
        print(f"frame {frame}: vols {vols} osm {osm}")


if __name__ == "__main__":
    main()
