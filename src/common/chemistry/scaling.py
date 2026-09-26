"""Dimensionless chemistry coordinates carried by AbstractTensor values.

The chemistry state stays in physical units at its public boundary. Newton
iterations operate on dimensionless variables and residuals so moles, joules,
kelvin, charge closure, and trace reaction extents do not compete merely
because their units have different numerical magnitudes.

This module contains no NumPy numerical path. Values enter through
``AbstractTensor`` and are promoted by the repository's ``Precision`` type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.tensors import AbstractTensor
from src.common.tensors.extended_precision import Precision


def _tensor(value: Any) -> AbstractTensor:
    if isinstance(value, AbstractTensor):
        return value
    return AbstractTensor.get_tensor(value)


def _wide(value: Any, limbs: int) -> Precision:
    if isinstance(value, Precision):
        if value.limbs != int(limbs):
            raise ValueError(
                f"wide chemistry value has {value.limbs} limbs, expected {limbs}"
            )
        return value
    return Precision.of(_tensor(value), int(limbs))


@dataclass(frozen=True)
class ChemistryScale:
    """Characteristic scales for one configured reduced chemistry system."""

    variable: AbstractTensor
    residual: AbstractTensor
    limbs: int = 2

    @classmethod
    def of(cls, variable: Any, residual: Any, *, limbs: int = 2) -> "ChemistryScale":
        if int(limbs) < 1:
            raise ValueError("chemistry precision requires at least one limb")
        variable_tensor = _tensor(variable)
        residual_tensor = _tensor(residual)
        if tuple(variable_tensor.shape) != tuple(residual_tensor.shape):
            raise ValueError(
                "variable and residual scales must have the same flattened shape"
            )
        if not bool((variable_tensor > 0.0).all().item()):
            raise ValueError("every chemistry variable scale must be positive")
        if not bool((residual_tensor > 0.0).all().item()):
            raise ValueError("every chemistry residual scale must be positive")
        return cls(variable_tensor, residual_tensor, int(limbs))

    def normalize_variables(self, physical: Any) -> Precision:
        """Promote physical variables and divide without collapsing limbs."""

        return Precision.of(_tensor(physical), self.limbs) / Precision.of(
            self.variable, self.limbs
        )

    def physical_variables(self, dimensionless: Any) -> Precision:
        """Return a wide physical value from a wide Newton coordinate."""

        return _wide(dimensionless, self.limbs) * Precision.of(
            self.variable, self.limbs
        )

    def normalize_residuals(self, physical: Any) -> Precision:
        """Condition residual equations without changing their roots."""

        return Precision.of(_tensor(physical), self.limbs) / Precision.of(
            self.residual, self.limbs
        )


@dataclass(frozen=True)
class ScaledChemistryState:
    """Dimensionless wide coordinates supplied to a Newton iteration."""

    variables: Precision
    residuals: Precision
    scale: ChemistryScale

    @classmethod
    def from_physical(
        cls, variables: Any, residuals: Any, scale: ChemistryScale,
    ) -> "ScaledChemistryState":
        return cls(
            scale.normalize_variables(variables),
            scale.normalize_residuals(residuals),
            scale,
        )
