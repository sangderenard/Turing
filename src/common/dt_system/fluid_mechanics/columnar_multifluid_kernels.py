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
    entity_b_x,
    entity_b_y,
    entity_b_velocity_x,
    entity_b_velocity_y,
    entity_c_x,
    entity_c_y,
    entity_c_velocity_x,
    entity_c_velocity_y,
    entity_cargo,
    entity_b_cargo,
    entity_c_cargo,
    food_store,
    nest_food,
    filter_reservoir,
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
    # Three ants deposit paired bands and follow the preceding ant's bands.
    # This cyclic colony rule is intentionally local and incomplete: trails,
    # repulsion and exploration decide the emergent path instead of a scripted
    # destination or the much slower display hue cycle.
    trail_a = ink_red + ink_yellow
    trail_b = ink_green + ink_cyan
    trail_c = ink_blue + ink_magenta
    cargo_a = entity_cargo.maximum(0.0).minimum(1.0)
    cargo_b = entity_b_cargo.maximum(0.0).minimum(1.0)
    cargo_c = entity_c_cargo.maximum(0.0).minimum(1.0)
    foraging_a = 1.0 - cargo_a
    foraging_b = 1.0 - cargo_b
    foraging_c = 1.0 - cargo_c
    nest_x = 5.0
    nest_y = 3.45

    # A new deterministic spatial hash is admitted every twelve and a half
    # managed seconds.  Its sparse fertile peaks grow into persistent food
    # entities; no host RNG or wall clock enters the state machine.
    # Lower spatial frequencies (was 12.9898/78.233/39.3467/11.135) widen the
    # fertile blobs from fine speckle into actual patches; a slower epoch
    # rate (was 0.08, ~12.5s cycle) and a smaller growth-rate constant (was
    # 0.032) make a patch take longer to grow back once it is eaten out.
    food_epoch = (next_time * 0.02).floor()
    food_hash_a = (
        column_x * 3.25 + column_y * 19.56 + food_epoch * 37.719
    ).sin()
    food_hash_b = (
        column_x * 9.84 - column_y * 2.78 + food_epoch * 19.913
    ).cos()
    food_fertility = (
        (food_hash_a * food_hash_b - 0.72).maximum(0.0) / 0.28
    )
    food_fertility = food_fertility * food_fertility
    growing_food = (
        food_store * (-0.003 * dt).exp() + 0.014 * dt * food_fertility
    ).maximum(0.0).minimum(1.0)

    sight_x = column_x - entity_x
    sight_y = column_y - entity_y
    sight_distance_squared = sight_x * sight_x + sight_y * sight_y
    sight_distance = (sight_distance_squared + 0.16).sqrt()
    visibility = (
        (-sight_distance_squared / (2.0 * 1.55 * 1.55)).exp()
        / (sight_distance_squared + 0.22)
    )
    signed_visibility = visibility * (
        1.20 * foraging_a * trail_c
        - 0.30 * trail_a + 1.65 * foraging_a * growing_food
        + 0.18 * angular_similarity
    )
    visibility_total = visibility.sum() + 1.0e-6
    steering_weight = signed_visibility / (sight_distance * visibility_total)
    steering_x = column_x * 0.0 + (sight_x * steering_weight).sum()
    steering_y = column_y * 0.0 + (sight_y * steering_weight).sum()
    separation_ab_x = entity_x - entity_b_x
    separation_ab_y = entity_y - entity_b_y
    separation_ac_x = entity_x - entity_c_x
    separation_ac_y = entity_y - entity_c_y
    separation_ab = (
        separation_ab_x * separation_ab_x
        + separation_ab_y * separation_ab_y + 0.30
    )
    separation_ac = (
        separation_ac_x * separation_ac_x
        + separation_ac_y * separation_ac_y + 0.30
    )
    nest_a_x = nest_x - entity_x
    nest_a_y = nest_y - entity_y
    nest_a_length = (nest_a_x * nest_a_x + nest_a_y * nest_a_y + 0.16).sqrt()
    acceleration_x = (
        2.35 * steering_x
        + 1.85 * cargo_a * nest_a_x / nest_a_length
        + 0.58 * (next_time * 1.71 + 0.20).cos()
        + 0.30 * separation_ab_x / separation_ab
        + 0.30 * separation_ac_x / separation_ac
        + 0.08 * (5.0 - entity_x) - 0.30 * entity_velocity_x
    )
    acceleration_y = (
        2.35 * steering_y
        + 1.85 * cargo_a * nest_a_y / nest_a_length
        + 0.58 * (next_time * 1.37 + 1.10).sin()
        + 0.30 * separation_ab_y / separation_ab
        + 0.30 * separation_ac_y / separation_ac
        + 0.08 * (3.5 - entity_y) - 0.30 * entity_velocity_y
    )
    next_entity_velocity_x = entity_velocity_x + acceleration_x * dt
    next_entity_velocity_y = entity_velocity_y + acceleration_y * dt
    next_entity_x = (
        entity_x + next_entity_velocity_x * dt
    ).maximum(0.65).minimum(9.35)
    next_entity_y = (
        entity_y + next_entity_velocity_y * dt
    ).maximum(0.65).minimum(6.35)

    sight_b_x = column_x - entity_b_x
    sight_b_y = column_y - entity_b_y
    sight_b_distance_squared = sight_b_x * sight_b_x + sight_b_y * sight_b_y
    sight_b_distance = (sight_b_distance_squared + 0.16).sqrt()
    visibility_b = (
        (-sight_b_distance_squared / (2.0 * 1.55 * 1.55)).exp()
        / (sight_b_distance_squared + 0.22)
    )
    signed_visibility_b = visibility_b * (
        1.20 * foraging_b * trail_a - 0.30 * trail_b
        + 1.65 * foraging_b * growing_food
        + 0.18 * (region_y * preferred_x - region_x * preferred_y)
    )
    visibility_b_total = visibility_b.sum() + 1.0e-6
    steering_b_weight = (
        signed_visibility_b / (sight_b_distance * visibility_b_total)
    )
    steering_b_x = column_x * 0.0 + (sight_b_x * steering_b_weight).sum()
    steering_b_y = column_y * 0.0 + (sight_b_y * steering_b_weight).sum()
    separation_bc_x = entity_b_x - entity_c_x
    separation_bc_y = entity_b_y - entity_c_y
    separation_bc = (
        separation_bc_x * separation_bc_x
        + separation_bc_y * separation_bc_y + 0.30
    )
    nest_b_x = nest_x - entity_b_x
    nest_b_y = nest_y - entity_b_y
    nest_b_length = (nest_b_x * nest_b_x + nest_b_y * nest_b_y + 0.16).sqrt()
    acceleration_b_x = (
        2.35 * steering_b_x
        + 1.85 * cargo_b * nest_b_x / nest_b_length
        + 0.58 * (next_time * 1.63 + 2.30).cos()
        - 0.30 * separation_ab_x / separation_ab
        + 0.30 * separation_bc_x / separation_bc
        + 0.08 * (5.0 - entity_b_x) - 0.30 * entity_b_velocity_x
    )
    acceleration_b_y = (
        2.35 * steering_b_y
        + 1.85 * cargo_b * nest_b_y / nest_b_length
        + 0.58 * (next_time * 1.43 + 2.80).sin()
        - 0.30 * separation_ab_y / separation_ab
        + 0.30 * separation_bc_y / separation_bc
        + 0.08 * (3.5 - entity_b_y) - 0.30 * entity_b_velocity_y
    )
    next_entity_b_velocity_x = entity_b_velocity_x + acceleration_b_x * dt
    next_entity_b_velocity_y = entity_b_velocity_y + acceleration_b_y * dt
    next_entity_b_x = (
        entity_b_x + next_entity_b_velocity_x * dt
    ).maximum(0.65).minimum(9.35)
    next_entity_b_y = (
        entity_b_y + next_entity_b_velocity_y * dt
    ).maximum(0.65).minimum(6.35)

    sight_c_x = column_x - entity_c_x
    sight_c_y = column_y - entity_c_y
    sight_c_distance_squared = sight_c_x * sight_c_x + sight_c_y * sight_c_y
    sight_c_distance = (sight_c_distance_squared + 0.16).sqrt()
    visibility_c = (
        (-sight_c_distance_squared / (2.0 * 1.55 * 1.55)).exp()
        / (sight_c_distance_squared + 0.22)
    )
    signed_visibility_c = visibility_c * (
        1.20 * foraging_c * trail_b - 0.30 * trail_c
        + 1.65 * foraging_c * growing_food
        - 0.18 * angular_similarity
    )
    visibility_c_total = visibility_c.sum() + 1.0e-6
    steering_c_weight = (
        signed_visibility_c / (sight_c_distance * visibility_c_total)
    )
    steering_c_x = column_x * 0.0 + (sight_c_x * steering_c_weight).sum()
    steering_c_y = column_y * 0.0 + (sight_c_y * steering_c_weight).sum()
    nest_c_x = nest_x - entity_c_x
    nest_c_y = nest_y - entity_c_y
    nest_c_length = (nest_c_x * nest_c_x + nest_c_y * nest_c_y + 0.16).sqrt()
    acceleration_c_x = (
        2.35 * steering_c_x
        + 1.85 * cargo_c * nest_c_x / nest_c_length
        + 0.58 * (next_time * 1.79 + 4.20).cos()
        - 0.30 * separation_ac_x / separation_ac
        - 0.30 * separation_bc_x / separation_bc
        + 0.08 * (5.0 - entity_c_x) - 0.30 * entity_c_velocity_x
    )
    acceleration_c_y = (
        2.35 * steering_c_y
        + 1.85 * cargo_c * nest_c_y / nest_c_length
        + 0.58 * (next_time * 1.31 + 5.00).sin()
        - 0.30 * separation_ac_y / separation_ac
        - 0.30 * separation_bc_y / separation_bc
        + 0.08 * (3.5 - entity_c_y) - 0.30 * entity_c_velocity_y
    )
    next_entity_c_velocity_x = entity_c_velocity_x + acceleration_c_x * dt
    next_entity_c_velocity_y = entity_c_velocity_y + acceleration_c_y * dt
    next_entity_c_x = (
        entity_c_x + next_entity_c_velocity_x * dt
    ).maximum(0.65).minimum(9.35)
    next_entity_c_y = (
        entity_c_y + next_entity_c_velocity_y * dt
    ).maximum(0.65).minimum(6.35)

    spectral_size = (
        0.42 * audio_low + 0.34 * audio_mid
        + 0.18 * audio_high + 0.35 * audio_level
    ).maximum(0.0).minimum(1.0)
    entity_half_extent = 0.18 + 0.12 * spectral_size
    distance_a_squared = (
        (column_x - next_entity_x) * (column_x - next_entity_x)
        + (column_y - next_entity_y) * (column_y - next_entity_y)
    )
    distance_b_squared = (
        (column_x - next_entity_b_x) * (column_x - next_entity_b_x)
        + (column_y - next_entity_b_y) * (column_y - next_entity_b_y)
    )
    distance_c_squared = (
        (column_x - next_entity_c_x) * (column_x - next_entity_c_x)
        + (column_y - next_entity_c_y) * (column_y - next_entity_c_y)
    )
    entity_a_interior = (
        (entity_half_extent - (column_x - next_entity_x).abs()).maximum(0.0)
        .minimum(
            (entity_half_extent - (column_y - next_entity_y).abs()).maximum(0.0)
        )
        / entity_half_extent
    )
    entity_b_interior = (
        (entity_half_extent - (column_x - next_entity_b_x).abs()).maximum(0.0)
        .minimum(
            (entity_half_extent - (column_y - next_entity_b_y).abs()).maximum(0.0)
        )
        / entity_half_extent
    )
    entity_c_interior = (
        (entity_half_extent - (column_x - next_entity_c_x).abs()).maximum(0.0)
        .minimum(
            (entity_half_extent - (column_y - next_entity_c_y).abs()).maximum(0.0)
        )
        / entity_half_extent
    )
    entity_interior = entity_a_interior.maximum(entity_b_interior)
    entity_interior = entity_interior.maximum(entity_c_interior)
    load_a = (-distance_a_squared / (2.0 * 0.78 * 0.78)).exp()
    load_b = (-distance_b_squared / (2.0 * 0.78 * 0.78)).exp()
    load_c = (-distance_c_squared / (2.0 * 0.78 * 0.78)).exp()
    load = (load_a + load_b + load_c).minimum(1.0)

    pickup_a_kernel = (-distance_a_squared / (2.0 * 0.30 * 0.30)).exp()
    pickup_b_kernel = (-distance_b_squared / (2.0 * 0.30 * 0.30)).exp()
    pickup_c_kernel = (-distance_c_squared / (2.0 * 0.30 * 0.30)).exp()
    food_near_a = (growing_food * pickup_a_kernel).sum()
    food_near_a = food_near_a / (pickup_a_kernel.sum() + 1.0e-6)
    food_near_b = (growing_food * pickup_b_kernel).sum()
    food_near_b = food_near_b / (pickup_b_kernel.sum() + 1.0e-6)
    food_near_c = (growing_food * pickup_c_kernel).sum()
    food_near_c = food_near_c / (pickup_c_kernel.sum() + 1.0e-6)
    pickup_a = foraging_a * food_near_a.minimum(0.7) * 1.35
    pickup_b = foraging_b * food_near_b.minimum(0.7) * 1.35
    pickup_c = foraging_c * food_near_c.minimum(0.7) * 1.35

    nest_distance_a = (
        (next_entity_x - nest_x) * (next_entity_x - nest_x)
        + (next_entity_y - nest_y) * (next_entity_y - nest_y)
    )
    nest_distance_b = (
        (next_entity_b_x - nest_x) * (next_entity_b_x - nest_x)
        + (next_entity_b_y - nest_y) * (next_entity_b_y - nest_y)
    )
    nest_distance_c = (
        (next_entity_c_x - nest_x) * (next_entity_c_x - nest_x)
        + (next_entity_c_y - nest_y) * (next_entity_c_y - nest_y)
    )
    nest_contact_a = (-nest_distance_a / (2.0 * 0.40 * 0.40)).exp()
    nest_contact_b = (-nest_distance_b / (2.0 * 0.40 * 0.40)).exp()
    nest_contact_c = (-nest_distance_c / (2.0 * 0.40 * 0.40)).exp()
    delivery_a = cargo_a * nest_contact_a * 2.4
    delivery_b = cargo_b * nest_contact_b * 2.4
    delivery_c = cargo_c * nest_contact_c * 2.4
    next_entity_cargo = (
        cargo_a + dt * (pickup_a - delivery_a)
    ).maximum(0.0).minimum(1.0)
    next_entity_b_cargo = (
        cargo_b + dt * (pickup_b - delivery_b)
    ).maximum(0.0).minimum(1.0)
    next_entity_c_cargo = (
        cargo_c + dt * (pickup_c - delivery_c)
    ).maximum(0.0).minimum(1.0)
    food_consumption = dt * (
        pickup_a_kernel * pickup_a
        + pickup_b_kernel * pickup_b
        + pickup_c_kernel * pickup_c
    )

    drain_x = column_x - 1.15
    utility_y = column_y - 5.75
    return_x = column_x - 8.85
    drain_distance = drain_x * drain_x + utility_y * utility_y
    return_distance = return_x * return_x + utility_y * utility_y
    drain_mask = (-drain_distance / (2.0 * 0.48 * 0.48)).exp()
    return_mask = (-return_distance / (2.0 * 0.58 * 0.58)).exp()
    next_food_store = (
        (growing_food - food_consumption).maximum(0.0)
        * (1.0 - 0.10 * dt * drain_mask)
    ).minimum(1.0)
    delivered_food = dt * (delivery_a + delivery_b + delivery_c)
    next_nest_food = (nest_food + delivered_food).maximum(0.0)
    # Pump aggressiveness: a higher cap and release rate move more filtered
    # fluid through faster (was 0.42 cap / 0.30 emission rate below), and a
    # larger surface-target coefficient makes the return jet actually push
    # the surface instead of barely nudging it (was 0.16).
    clean_release = filter_reservoir.maximum(0.0).minimum(0.65)
    target = (
        -0.42 * load - 0.22 * entity_interior * entity_interior
        + 0.30 * clean_release * return_mask
    )
    # Damping lowered (was 4.6) so the surface actually oscillates instead
    # of critically damping back to rest almost immediately.
    acceleration = (
        20.0 * (target - displacement) - 2.6 * displacement_velocity
    )
    next_velocity = displacement_velocity + acceleration * dt
    next_displacement = displacement + next_velocity * dt
    surface = rest_surface + next_displacement
    # Both the divisor (was 5.0) and the color coefficients below (was
    # 27/34/18/21/16/15) were small enough that real displacement barely
    # moved the rendered color -- height now saturates over a narrower
    # displacement range and pushes color much harder once it does.
    height = ((surface - 0.5) / 2.0).maximum(0.0).minimum(1.0)
    compression = (-next_displacement / 0.42).maximum(0.0).minimum(1.0)
    motion = next_velocity.abs().minimum(1.0)

    # A synthesized flow-direction field: each column's velocity is the
    # inverse-distance-weighted blend of the three ants' own velocities,
    # since the surface's own vertical spring velocity carries no x/y
    # direction to show. Tinting by this is the closest a scalar RGB raster
    # gets to drawing velocity vectors without literal arrow geometry.
    flow_weight_a = 1.0 / (distance_a_squared + 0.35)
    flow_weight_b = 1.0 / (distance_b_squared + 0.35)
    flow_weight_c = 1.0 / (distance_c_squared + 0.35)
    flow_weight_total = flow_weight_a + flow_weight_b + flow_weight_c
    flow_x = (
        flow_weight_a * next_entity_velocity_x
        + flow_weight_b * next_entity_b_velocity_x
        + flow_weight_c * next_entity_c_velocity_x
    ) / flow_weight_total
    flow_y = (
        flow_weight_a * next_entity_velocity_y
        + flow_weight_b * next_entity_b_velocity_y
        + flow_weight_c * next_entity_c_velocity_y
    ) / flow_weight_total

    base_red = (
        186.0 + 50.0 * height - 60.0 * compression + 54.0 * load
        + 16.0 * flow_x
    ).maximum(0.0).minimum(255.0)
    base_green = (
        220.0 + 35.0 * height - 40.0 * compression + 30.0 * load
        + 26.0 * motion
    ).maximum(0.0).minimum(255.0)
    base_blue = (
        232.0 + 30.0 * height + 28.0 * compression + 20.0 * load
        + 16.0 * flow_y
    ).maximum(0.0).minimum(255.0)

    # Paired fields are the ants' pheromone vocabulary.  The narrow band is
    # the fresh trail and the wider companion is its smoothly overlapping
    # halo; unequal evaporation leaves useful gradients behind moving ants.
    source_red = (-distance_a_squared / (2.0 * 0.24 * 0.24)).exp()
    source_yellow = (-distance_a_squared / (2.0 * 0.38 * 0.38)).exp()
    source_green = (-distance_b_squared / (2.0 * 0.24 * 0.24)).exp()
    source_cyan = (-distance_b_squared / (2.0 * 0.38 * 0.38)).exp()
    source_blue = (-distance_c_squared / (2.0 * 0.24 * 0.24)).exp()
    source_magenta = (-distance_c_squared / (2.0 * 0.38 * 0.38)).exp()
    pulse_a = 0.78 + 0.22 * (next_time * 2.11).sin()
    pulse_b = 0.78 + 0.22 * (next_time * 1.91 + 2.1).sin()
    pulse_c = 0.78 + 0.22 * (next_time * 2.27 + 4.2).sin()
    core_decay = (-0.095 * dt).exp()
    halo_decay = (-0.072 * dt).exp()
    core_gain = 3.2 * dt
    halo_gain = 2.2 * dt
    next_ink_red = (
        ink_red * core_decay + core_gain * source_red * pulse_a
    ).minimum(1.0)
    next_ink_yellow = (
        ink_yellow * halo_decay + halo_gain * source_yellow * pulse_a
    ).minimum(1.0)
    next_ink_green = (
        ink_green * core_decay + core_gain * source_green * pulse_b
    ).minimum(1.0)
    next_ink_cyan = (
        ink_cyan * halo_decay + halo_gain * source_cyan * pulse_b
    ).minimum(1.0)
    next_ink_blue = (
        ink_blue * core_decay + core_gain * source_blue * pulse_c
    ).minimum(1.0)
    next_ink_magenta = (
        ink_magenta * halo_decay + halo_gain * source_magenta * pulse_c
    ).minimum(1.0)
    drain_fraction = (0.12 * dt * drain_mask).minimum(0.08)
    drainable_material = (
        next_ink_red + next_ink_yellow + next_ink_green
        + next_ink_cyan + next_ink_blue + next_ink_magenta
    )
    drained_material = (drainable_material * drain_mask).sum()
    drained_material = (
        0.12 * dt * drained_material / (drain_mask.sum() + 1.0e-6)
    )
    drain_retention = 1.0 - drain_fraction
    next_ink_red = next_ink_red * drain_retention
    next_ink_yellow = next_ink_yellow * drain_retention
    next_ink_green = next_ink_green * drain_retention
    next_ink_cyan = next_ink_cyan * drain_retention
    next_ink_blue = next_ink_blue * drain_retention
    next_ink_magenta = next_ink_magenta * drain_retention
    clean_emission = 0.70 * dt * clean_release
    next_filter_reservoir = (
        filter_reservoir + drained_material - clean_emission
    ).maximum(0.0)
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
    ink_retention = 1.0 - ink_alpha
    red = base_red * ink_retention + ink_color_red * ink_alpha
    green = base_green * ink_retention + ink_color_green * ink_alpha
    blue = base_blue * ink_retention + ink_color_blue * ink_alpha

    food_alpha = next_food_store.minimum(0.76)
    food_retention = 1.0 - food_alpha
    red = red * food_retention + 226.0 * food_alpha
    green = green * food_retention + 181.0 * food_alpha
    blue = blue * food_retention + 62.0 * food_alpha
    nest_field_x = column_x - nest_x
    nest_field_y = column_y - nest_y
    nest_field_distance = (
        nest_field_x * nest_field_x + nest_field_y * nest_field_y
    )
    nest_body = (-nest_field_distance / (2.0 * 0.34 * 0.34)).exp()
    nest_glow = next_nest_food.minimum(1.0)
    nest_red = 102.0 + 58.0 * nest_glow
    nest_green = 72.0 + 42.0 * nest_glow
    nest_blue = 48.0 + 24.0 * nest_glow
    nest_retention = 1.0 - nest_body
    red = red * nest_retention + nest_red * nest_body
    green = green * nest_retention + nest_green * nest_body
    blue = blue * nest_retention + nest_blue * nest_body

    clean_alpha = (return_mask * clean_release).minimum(0.80)
    clean_retention = 1.0 - clean_alpha
    red = red * clean_retention + 218.0 * clean_alpha
    green = green * clean_retention + 249.0 * clean_alpha
    blue = blue * clean_retention + 255.0 * clean_alpha
    drain_body = (-drain_distance / (2.0 * 0.23 * 0.23)).exp()
    return_body = (-return_distance / (2.0 * 0.23 * 0.23)).exp()
    drain_body_retention = 1.0 - drain_body
    return_body_retention = 1.0 - return_body
    red = red * drain_body_retention + 53.0 * drain_body
    green = green * drain_body_retention + 83.0 * drain_body
    blue = blue * drain_body_retention + 103.0 * drain_body
    red = red * return_body_retention + 205.0 * return_body
    green = green * return_body_retention + 245.0 * return_body
    blue = blue * return_body_retention + 252.0 * return_body
    entity_glow = entity_interior * entity_interior
    entity_retention = 1.0 - entity_glow
    red = red * entity_retention + 245.0 * entity_glow
    green = green * entity_retention + 252.0 * entity_glow
    blue = blue * entity_retention + 255.0 * entity_glow
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
        next_entity_b_x,
        next_entity_b_y,
        next_entity_b_velocity_x,
        next_entity_b_velocity_y,
        next_entity_c_x,
        next_entity_c_y,
        next_entity_c_velocity_x,
        next_entity_c_velocity_y,
        next_entity_cargo,
        next_entity_b_cargo,
        next_entity_c_cargo,
        next_food_store,
        next_nest_food,
        next_filter_reservoir,
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
