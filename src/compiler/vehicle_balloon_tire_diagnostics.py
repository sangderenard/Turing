"""Read-only diagnostics for the authoritative balloon-tire state.

The observer uses the repository's AbstractTensor cotangent
Laplace--Beltrami operator on the actual deformed skin.  It never contributes
force, damping, or contact, so diagnostic backend choices cannot change the
vehicle solution.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.common.tensors import AbstractTensor
from src.common.tensors.riemann import abstract_mesh_laplace

from .vehicle_balloon_tire import BalloonTireTopology


def balloon_tire_state_diagnostics(
    state: Sequence[float],
    reference_state: Sequence[float],
    topology: BalloonTireTopology,
    vertex_mass_kg: float,
) -> dict:
    """Classify energy localization on one six-lane-per-vertex tire state."""

    vertex_count = len(topology.rest_positions)
    vertices = np.asarray([
        [state[6 * vertex + axis] for axis in range(3)]
        for vertex in range(vertex_count)
    ])
    displacement = np.asarray([
        [state[6 * vertex + axis] - reference_state[6 * vertex + axis]
         for axis in range(3)]
        for vertex in range(vertex_count)
    ])
    triangles = np.asarray(topology.faces)
    laplacian = np.column_stack([
        np.asarray(abstract_mesh_laplace(
            AbstractTensor.tensor(vertices), triangles,
            AbstractTensor.tensor(displacement[:, axis]),
        ).tolist())
        for axis in range(3)
    ])
    local = np.sum(laplacian * laplacian, axis=1)
    laplacian_energy = float(np.sum(local))
    displacement_energy = float(np.sum(displacement * displacement))

    section_count = topology.section_segments
    circumference_count = topology.circumferential_segments
    local_displacement = np.empty_like(displacement)
    for iu in range(circumference_count):
        angle = 2.0 * math.pi * iu / circumference_count
        ca, sa = math.cos(angle), math.sin(angle)
        rows = slice(iu * section_count, (iu + 1) * section_count)
        local_displacement[rows, 0] = (
            ca * displacement[rows, 0] + sa * displacement[rows, 1]
        )
        local_displacement[rows, 1] = (
            -sa * displacement[rows, 0] + ca * displacement[rows, 1]
        )
        local_displacement[rows, 2] = displacement[rows, 2]
    section_field = local_displacement.reshape(
        circumference_count, section_count, 3,
    )
    axisymmetric = np.mean(section_field, axis=0, keepdims=True)
    asymmetric_energy = float(np.sum((section_field - axisymmetric) ** 2))
    asymmetry_fraction = asymmetric_energy / max(displacement_energy, 1e-30)
    top_concentration = float(sum(sorted(local.tolist(), reverse=True)[:5])) / max(
        laplacian_energy, 1e-30,
    )
    mode_class = (
        "axisymmetric-section-mode" if asymmetry_fraction < 0.1
        else "localized-high-frequency-mode" if top_concentration > 0.25
        else "distributed-nonaxisymmetric-mode"
    )
    kinetic = 0.5 * vertex_mass_kg * sum(
        state[6 * vertex + axis] ** 2
        for vertex in range(vertex_count) for axis in range(3, 6)
    )
    return {
        "observer": "abstract-tensor-cotangent-laplace-beltrami",
        "kinetic_energy_j": kinetic,
        "laplace_beltrami_energy": laplacian_energy,
        "laplacian_roughness": laplacian_energy / max(displacement_energy, 1e-30),
        "circumferential_asymmetry_fraction": asymmetry_fraction,
        "top_five_laplacian_concentration": top_concentration,
        "mode_class": mode_class,
        "top_vertices": [
            {"vertex": index, "local_laplacian_energy": score}
            for index, score in sorted(
                enumerate(local.tolist()), key=lambda item: item[1], reverse=True,
            )[:5]
        ],
    }


__all__ = ["balloon_tire_state_diagnostics"]
