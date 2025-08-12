from __future__ import annotations
from typing import Iterable
import math
from .units import R as RGAS, EPS
from tqdm.auto import tqdm  # type: ignore


def assert_nonneg(cells, bath, species: Iterable[str]):
    for c in tqdm(cells, desc="cells", leave=False):
        assert c.V >= 0.0, "negative volume"
        for sp in tqdm(species, desc="species", leave=False):
            assert c.n.get(sp, 0.0) >= -EPS, f"negative amount in cell {sp}"
            for o in tqdm(getattr(c, "organelles", []), desc="organelles", leave=False):
                assert o.n.get(sp, 0.0) >= -EPS, f"negative amount in organelle {sp}"
    assert bath.V >= 0.0, "negative bath volume"
    for sp in tqdm(species, desc="bath species", leave=False):
        assert bath.n.get(sp, 0.0) >= -EPS, f"negative amount in bath {sp}"


def assert_mass_conserved(cells, bath, species: Iterable[str], totals: dict[str, float], tol: float = 1e-6):
    for sp in tqdm(species, desc="species", leave=False):
        total = bath.n.get(sp, 0.0)
        for c in tqdm(cells, desc="cells", leave=False):
            total += c.n.get(sp, 0.0)
            for o in tqdm(getattr(c, "organelles", []), desc="organelles", leave=False):
                total += o.n.get(sp, 0.0)
        assert abs(total - totals.get(sp, total)) < tol, f"mass not conserved for {sp}"


def assert_passive_no_energy(cell, bath, dS_cell: dict, Cint: dict, Cext: dict, species: Iterable[str], T: float, Rgas: float = RGAS):
    for sp in tqdm(species, desc="species", leave=False):
        flux = dS_cell.get(sp, 0.0)
        if flux == 0.0:
            continue
        Ci = Cint.get(sp, 0.0)
        Ce = Cext.get(sp, 0.0)
        if Ci <= 0 or Ce <= 0:
            continue
        mu_diff = Rgas * T * math.log((Ci + EPS)/(Ce + EPS))
        assert flux * mu_diff <= EPS, f"passive flux produced energy for {sp}"
