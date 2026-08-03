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
    entity_x,
    entity_y,
    entity_velocity_x,
    entity_velocity_y,
    managed_time,
    dt,
    audio_low,
    audio_mid,
    audio_high,
    audio_level,
    ink_red,
    ink_yellow,
    ink_green,
    ink_cyan,
    ink_blue,
    ink_magenta,
):
    """One Python-owned managed tick and its three RGB preview planes."""

    next_time = managed_time + dt
    preferred_x = audio_low - audio_high
    preferred_y = 2.0 * audio_mid - audio_low - audio_high
    preferred_length = (
        preferred_x * preferred_x + preferred_y * preferred_y + 1.0e-5
    ).sqrt()
    preferred_x = preferred_x / preferred_length
    preferred_y = preferred_y / preferred_length
    region_phase = (
        column_x * 0.61 + column_y * 0.83
        + (column_x * 0.37 - column_y * 0.29).sin() * 0.72
    )
    region_x = region_phase.cos()
    region_y = region_phase.sin()
    angular_similarity = region_x * preferred_x + region_y * preferred_y
    sight_x = column_x - entity_x
    sight_y = column_y - entity_y
    sight_distance_squared = sight_x * sight_x + sight_y * sight_y
    sight_distance = (sight_distance_squared + 0.12).sqrt()
    visibility = (
        (-sight_distance_squared / (2.0 * 2.35 * 2.35)).exp()
        / (sight_distance_squared + 0.18)
    )
    signed_visibility = visibility * angular_similarity
    visibility_total = visibility.sum() + 1.0e-6
    steering_x = (
        (sight_x / sight_distance) * signed_visibility
    ).sum() / visibility_total
    steering_y = (
        (sight_y / sight_distance) * signed_visibility
    ).sum() / visibility_total
    steering_x = column_x * 0.0 + steering_x
    steering_y = column_y * 0.0 + steering_y
    acceleration_x = (
        1.85 * steering_x + 0.10 * (5.0 - entity_x)
        - 0.72 * entity_velocity_x
    )
    acceleration_y = (
        1.85 * steering_y + 0.10 * (3.5 - entity_y)
        - 0.72 * entity_velocity_y
    )
    next_entity_velocity_x = entity_velocity_x + acceleration_x * dt
    next_entity_velocity_y = entity_velocity_y + acceleration_y * dt
    next_entity_x = (
        entity_x + next_entity_velocity_x * dt
    ).maximum(0.65).minimum(9.35)
    next_entity_y = (
        entity_y + next_entity_velocity_y * dt
    ).maximum(0.65).minimum(6.35)
    player_x = next_entity_x
    player_y = next_entity_y
    spectral_size = (
        0.42 * audio_low + 0.34 * audio_mid
        + 0.18 * audio_high + 0.35 * audio_level
    ).maximum(0.0).minimum(1.0)
    entity_half_extent = 0.32 + 0.30 * spectral_size
    distance_squared = (
        (column_x - player_x) * (column_x - player_x)
        + (column_y - player_y) * (column_y - player_y)
    )
    entity_interior = (
        (entity_half_extent - (column_x - player_x).abs()).maximum(0.0)
        .minimum(
            (entity_half_extent - (column_y - player_y).abs()).maximum(0.0)
        )
        / entity_half_extent
    )
    load = (-distance_squared / (2.0 * 1.35 * 1.35)).exp()
    target = -0.42 * load - 0.22 * entity_interior * entity_interior
    acceleration = (
        20.0 * (target - displacement) - 8.0 * displacement_velocity
    )
    next_velocity = displacement_velocity + acceleration * dt
    next_displacement = displacement + next_velocity * dt
    surface = rest_surface + next_displacement
    height = ((surface - 0.5) / 5.0).maximum(0.0).minimum(1.0)
    compression = (-next_displacement / 0.42).maximum(0.0).minimum(1.0)
    motion = next_velocity.abs().minimum(1.0)

    base_red = (
        186.0 + 27.0 * height - 34.0 * compression + 54.0 * load
    ).maximum(0.0).minimum(255.0)
    base_green = (
        220.0 + 18.0 * height - 21.0 * compression + 30.0 * load
        + 8.0 * motion
    ).maximum(0.0).minimum(255.0)
    base_blue = (
        232.0 + 16.0 * height + 15.0 * compression + 20.0 * load
    ).maximum(0.0).minimum(255.0)

    hue = next_time * 0.42
    source_red = (-distance_squared / (2.0 * 0.44 * 0.44)).exp()
    source_yellow = (-distance_squared / (2.0 * 0.48 * 0.48)).exp()
    source_green = (-distance_squared / (2.0 * 0.52 * 0.52)).exp()
    source_cyan = (-distance_squared / (2.0 * 0.56 * 0.56)).exp()
    source_blue = (-distance_squared / (2.0 * 0.60 * 0.60)).exp()
    source_magenta = (-distance_squared / (2.0 * 0.64 * 0.64)).exp()
    weight_red = hue.cos().maximum(0.0)
    weight_yellow = (hue - 1.0471975511965976).cos().maximum(0.0)
    weight_green = (hue - 2.0943951023931953).cos().maximum(0.0)
    weight_cyan = (hue - 3.141592653589793).cos().maximum(0.0)
    weight_blue = (hue - 4.1887902047863905).cos().maximum(0.0)
    weight_magenta = (hue - 5.235987755982989).cos().maximum(0.0)
    next_ink_red = (
        ink_red * (-0.050 * dt).exp() + 2.8 * dt * source_red * weight_red
    ).minimum(1.0)
    next_ink_yellow = (
        ink_yellow * (-0.054 * dt).exp()
        + 2.8 * dt * source_yellow * weight_yellow
    ).minimum(1.0)
    next_ink_green = (
        ink_green * (-0.058 * dt).exp() + 2.8 * dt * source_green * weight_green
    ).minimum(1.0)
    next_ink_cyan = (
        ink_cyan * (-0.062 * dt).exp() + 2.8 * dt * source_cyan * weight_cyan
    ).minimum(1.0)
    next_ink_blue = (
        ink_blue * (-0.066 * dt).exp() + 2.8 * dt * source_blue * weight_blue
    ).minimum(1.0)
    next_ink_magenta = (
        ink_magenta * (-0.070 * dt).exp()
        + 2.8 * dt * source_magenta * weight_magenta
    ).minimum(1.0)
    ink_total = (
        next_ink_red + next_ink_yellow + next_ink_green
        + next_ink_cyan + next_ink_blue + next_ink_magenta
    ).maximum(1.0e-6)
    ink_alpha = ink_total.minimum(0.88)
    ink_color_red = 255.0 * (
        next_ink_red + next_ink_yellow + next_ink_magenta
    ) / ink_total
    ink_color_green = 255.0 * (
        next_ink_yellow + next_ink_green + next_ink_cyan
    ) / ink_total
    ink_color_blue = 255.0 * (
        next_ink_cyan + next_ink_blue + next_ink_magenta
    ) / ink_total
    red = base_red * (1.0 - ink_alpha) + ink_color_red * ink_alpha
    green = base_green * (1.0 - ink_alpha) + ink_color_green * ink_alpha
    blue = base_blue * (1.0 - ink_alpha) + ink_color_blue * ink_alpha
    entity_glow = entity_interior * entity_interior
    red = red * (1.0 - entity_glow) + 245.0 * entity_glow
    green = green * (1.0 - entity_glow) + 252.0 * entity_glow
    blue = blue * (1.0 - entity_glow) + 255.0 * entity_glow
    return (
        red,
        green,
        blue,
        next_displacement,
        next_velocity,
        next_entity_x,
        next_entity_y,
        next_entity_velocity_x,
        next_entity_velocity_y,
        next_time,
        next_ink_red,
        next_ink_yellow,
        next_ink_green,
        next_ink_cyan,
        next_ink_blue,
        next_ink_magenta,
    )


__all__ = [
    "advance_columnar_spring_from_load",
    "advance_columnar_surface_spring_local",
    "columnar_multifluid_rgb_step",
]
