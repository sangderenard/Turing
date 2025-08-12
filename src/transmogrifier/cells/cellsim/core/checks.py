from __future__ import annotations
from typing import Iterable
import math
from .units import R as RGAS, EPS


def assert_nonneg(cells, bath, species: Iterable[str]):
    for c in cells:
        assert c.V >= 0.0, "negative volume"
        for sp in species:
            assert c.n.get(sp, 0.0) >= -EPS, f"negative amount in cell {sp}"
            for o in getattr(c, "organelles", []):
                assert o.n.get(sp, 0.0) >= -EPS, f"negative amount in organelle {sp}"
    assert bath.V >= 0.0, "negative bath volume"
    for sp in species:
        assert bath.n.get(sp, 0.0) >= -EPS, f"negative amount in bath {sp}"


def assert_mass_conserved(cells, bath, species: Iterable[str], totals: dict[str, float], tol: float = 1e-6):
    for sp in species:
        total = bath.n.get(sp, 0.0)
        for c in cells:
            total += c.n.get(sp, 0.0)
            for o in getattr(c, "organelles", []):
                total += o.n.get(sp, 0.0)
        assert abs(total - totals.get(sp, total)) < tol, f"mass not conserved for {sp}"


def assert_passive_no_energy(cell, bath, dS_cell: dict, Cint: dict, Cext: dict, species: Iterable[str], T: float, Rgas: float = RGAS):
    for sp in species:
        flux = dS_cell.get(sp, 0.0)
        if flux == 0.0:
            continue
        Ci = Cint.get(sp, 0.0)
        Ce = Cext.get(sp, 0.0)
        if Ci <= 0 or Ce <= 0:
            continue
        mu_diff = Rgas * T * math.log((Ci + EPS)/(Ce + EPS))
        assert flux * mu_diff <= EPS, f"passive flux produced energy for {sp}"
