"""Default biophysical constants for cellsim.

These values target typical eukaryotic cells but can be overridden
through presets for other cell types.
"""
from __future__ import annotations
from dataclasses import dataclass

# Amount of impermeant solute (moles) per unit of abstract data.
# This provides a stable osmotic load for each data-carrying organelle.
SALINITY_PER_DATA_UNIT: float = 1e-3

# Baseline hydraulic permeability of the plasma membrane (m/Pa/s).
DEFAULT_LP0: float = 1e-12

# Effective surface elastic modulus of the membrane (Pa).
DEFAULT_ELASTIC_K: float = 1e5


@dataclass(frozen=True)
class Preset:
    """Biophysical parameter bundle for a cell type."""
    salinity_per_data_unit: float = SALINITY_PER_DATA_UNIT
    elastic_k: float = DEFAULT_ELASTIC_K
    Lp0: float = DEFAULT_LP0


# Predefined parameter sets keyed by shorthand names.  Additional
# presets may be added by callers.
PRESETS: dict[str, Preset] = {
    "eukaryote": Preset(),
}


def apply_preset(cell, preset: str = "eukaryote") -> None:
    """Apply the named preset parameters to ``cell`` in-place."""
    cfg = PRESETS.get(preset)
    if cfg is None:
        raise KeyError(f"Unknown preset '{preset}'")
    cell.elastic_k = cfg.elastic_k
    cell.Lp0 = cfg.Lp0
