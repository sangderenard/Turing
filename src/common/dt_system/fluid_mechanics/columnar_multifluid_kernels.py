"""Shared Python transitions for the native and recompiled columnar demo.

These are ordinary Python functions over AbstractTensor-compatible values.
The live state machine calls the same spring transition that the Python AST
compiler recompiles for WebAssembly; there is no browser-language copy.
"""

from __future__ import annotations


def advance_columnar_spring_from_load(
    displacement,
    displacement_velocity,
    load,
    spring_stiffness,
    spring_damping,
    load_depth,
    load_radius,
    dt,
):
    """Advance independent column springs from an already classified load."""

    target = -load_depth * load
    acceleration = (
        spring_stiffness * (target - displacement.reshape((-1,)))
        - spring_damping * displacement_velocity.reshape((-1,))
    )
    next_velocity = displacement_velocity.reshape((-1,)) + acceleration * dt
    next_displacement = displacement.reshape((-1,)) + next_velocity * dt
    return next_displacement, next_velocity, load


def advance_columnar_surface_spring_local(
    column_centroid,
    displacement,
    displacement_velocity,
    player_centroid,
    spring_stiffness,
    spring_damping,
    load_depth,
    load_radius,
    dt,
):
    """Classify all player loads, then advance the parallel spring sheet."""

    delta = (
        column_centroid.reshape((-1, 1, 2))
        - player_centroid[..., :2].reshape((1, -1, 2))
    )
    distance_squared = (delta * delta).sum(dim=-1)
    load = (
        -distance_squared / (2.0 * load_radius * load_radius)
    ).exp().max(dim=-1)
    return advance_columnar_spring_from_load(
        displacement,
        displacement_velocity,
        load,
        spring_stiffness,
        spring_damping,
        load_depth,
        load_radius,
        dt,
    )


def columnar_multifluid_rgb_step(
    column_x,
    column_y,
    rest_surface,
    displacement,
    displacement_velocity,
    managed_time,
    dt,
):
    """One Python-owned managed tick and its three RGB preview planes."""

    next_time = managed_time + dt
    player_x = 5.0 + 3.15 * (next_time * 0.72).sin()
    player_y = 3.5 + 2.05 * (next_time * 1.07 + 1.5707963267948966).sin()
    distance_squared = (
        (column_x - player_x) * (column_x - player_x)
        + (column_y - player_y) * (column_y - player_y)
    )
    load = (-distance_squared / (2.0 * 1.35 * 1.35)).exp()
    target = -0.42 * load
    acceleration = (
        20.0 * (target - displacement) - 8.0 * displacement_velocity
    )
    next_velocity = displacement_velocity + acceleration * dt
    next_displacement = displacement + next_velocity * dt
    surface = rest_surface + next_displacement
    height = ((surface - 0.5) / 5.0).maximum(0.0).minimum(1.0)
    compression = (-next_displacement / 0.42).maximum(0.0).minimum(1.0)
    motion = next_velocity.abs().minimum(1.0)

    red = (
        186.0 + 27.0 * height - 34.0 * compression + 54.0 * load
    ).maximum(0.0).minimum(255.0)
    green = (
        220.0 + 18.0 * height - 21.0 * compression + 30.0 * load
        + 8.0 * motion
    ).maximum(0.0).minimum(255.0)
    blue = (
        232.0 + 16.0 * height + 15.0 * compression + 20.0 * load
    ).maximum(0.0).minimum(255.0)
    return (
        red,
        green,
        blue,
        next_displacement,
        next_velocity,
        next_time,
    )


__all__ = [
    "advance_columnar_spring_from_load",
    "advance_columnar_surface_spring_local",
    "columnar_multifluid_rgb_step",
]
