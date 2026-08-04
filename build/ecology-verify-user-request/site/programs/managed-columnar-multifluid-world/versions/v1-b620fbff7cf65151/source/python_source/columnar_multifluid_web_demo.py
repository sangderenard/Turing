TURING_PAGE = {'entrypoint': 'columnar_multifluid_rgb_step', 'presentation_entrypoint': 'columnar_multifluid_present', 'title': 'Managed Columnar Multifluid World', 'slug': 'managed-columnar-multifluid-world', 'width': 384, 'height': 268, 'probe_size': 16, 'feeds': {'column_x': 0.5, 'column_y': 0.5, 'rest_surface': 1.0, 'displacement': 0.0, 'displacement_velocity': 0.0, 'entity_x': 2.6, 'entity_y': 2.0, 'entity_velocity_x': 0.42, 'entity_velocity_y': 0.16, 'entity_b_x': 5.1, 'entity_b_y': 4.8, 'entity_b_velocity_x': -0.34, 'entity_b_velocity_y': 0.27, 'entity_c_x': 7.5, 'entity_c_y': 2.8, 'entity_c_velocity_x': 0.18, 'entity_c_velocity_y': -0.38, 'entity_cargo': 0.0, 'entity_b_cargo': 0.0, 'entity_c_cargo': 0.0, 'food_store': 0.0, 'nest_food': 0.0, 'filter_reservoir': 0.0, 'managed_time': 0.0, 'dt': 0.025, 'audio_low': 0.0, 'audio_mid': 0.0, 'audio_high': 0.0, 'audio_level': 0.0, 'ink_red': 0.0, 'ink_yellow': 0.0, 'ink_green': 0.0, 'ink_cyan': 0.0, 'ink_blue': 0.0, 'ink_magenta': 0.0}, 'feed_expressions': {'column_x': '(x + 0.5) * 10.0 / w', 'column_y': '(y + 0.5) * 7.0 / h', 'rest_surface': '1.15 + 3.1 * Math.exp(-16.0 * (((x + 0.5) / w - 0.62) ** 2 + ((y + 0.5) / h - 0.52) ** 2))', 'displacement': '0.0', 'displacement_velocity': '0.0', 'entity_x': '2.6', 'entity_y': '2.0', 'entity_velocity_x': '0.42', 'entity_velocity_y': '0.16', 'entity_b_x': '5.1', 'entity_b_y': '4.8', 'entity_b_velocity_x': '-0.34', 'entity_b_velocity_y': '0.27', 'entity_c_x': '7.5', 'entity_c_y': '2.8', 'entity_c_velocity_x': '0.18', 'entity_c_velocity_y': '-0.38', 'entity_cargo': '0.0', 'entity_b_cargo': '0.0', 'entity_c_cargo': '0.0', 'food_store': '0.0', 'nest_food': '0.0', 'filter_reservoir': '0.0', 'managed_time': '0.0', 'dt': '0.025', 'audio_low': "window.TuringAudioRuntime ? window.TuringAudioRuntime.feature('audio_low') : 0.0", 'audio_mid': "window.TuringAudioRuntime ? window.TuringAudioRuntime.feature('audio_mid') : 0.0", 'audio_high': "window.TuringAudioRuntime ? window.TuringAudioRuntime.feature('audio_high') : 0.0", 'audio_level': "window.TuringAudioRuntime ? window.TuringAudioRuntime.feature('audio_level') : 0.0", 'ink_red': '0.0', 'ink_yellow': '0.0', 'ink_green': '0.0', 'ink_cyan': '0.0', 'ink_blue': '0.0', 'ink_magenta': '0.0'}, 'state_feedback': {'displacement': 'next_displacement', 'displacement_velocity': 'next_velocity', 'entity_x': 'next_entity_x', 'entity_y': 'next_entity_y', 'entity_velocity_x': 'next_entity_velocity_x', 'entity_velocity_y': 'next_entity_velocity_y', 'entity_b_x': 'next_entity_b_x', 'entity_b_y': 'next_entity_b_y', 'entity_b_velocity_x': 'next_entity_b_velocity_x', 'entity_b_velocity_y': 'next_entity_b_velocity_y', 'entity_c_x': 'next_entity_c_x', 'entity_c_y': 'next_entity_c_y', 'entity_c_velocity_x': 'next_entity_c_velocity_x', 'entity_c_velocity_y': 'next_entity_c_velocity_y', 'entity_cargo': 'next_entity_cargo', 'entity_b_cargo': 'next_entity_b_cargo', 'entity_c_cargo': 'next_entity_c_cargo', 'food_store': 'next_food_store', 'nest_food': 'next_nest_food', 'filter_reservoir': 'next_filter_reservoir', 'managed_time': 'next_time', 'ink_red': 'next_ink_red', 'ink_yellow': 'next_ink_yellow', 'ink_green': 'next_ink_green', 'ink_cyan': 'next_ink_cyan', 'ink_blue': 'next_ink_blue', 'ink_magenta': 'next_ink_magenta'}, 'render_fps': 30.0, 'autostart': True, 'backend': 'c', 'remove_loops': True, 'audio': {'generator': 'src.common.dt_system.fluid_mechanics.columnar_multifluid_audio:synthesize_columnar_audio', 'arguments': {'duration': 8.0, 'sample_rate': 24000, 'feature_fps': 30}, 'managed_time_output': 'next_time', 'pan_output': 'next_entity_x', 'pan_range': [0.0, 10.0]}}

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
    nest_x = 5.0
    nest_y = 3.45

    # A new deterministic spatial hash is admitted every twelve and a half
    # managed seconds.  Its sparse fertile peaks grow into persistent food
    # entities; no host RNG or wall clock enters the state machine.
    food_epoch = (next_time * 0.08).floor()
    food_hash_a = (
        column_x * 12.9898 + column_y * 78.233 + food_epoch * 37.719
    ).sin()
    food_hash_b = (
        column_x * 39.3467 - column_y * 11.135 + food_epoch * 19.913
    ).cos()
    food_fertility = (
        (food_hash_a * food_hash_b - 0.72).maximum(0.0) / 0.28
    )
    food_fertility = food_fertility * food_fertility
    growing_food = (
        food_store * (-0.003 * dt).exp() + 0.032 * dt * food_fertility
    ).maximum(0.0).minimum(1.0)

    sight_x = column_x - entity_x
    sight_y = column_y - entity_y
    sight_distance_squared = sight_x * sight_x + sight_y * sight_y
    sight_distance = (sight_distance_squared + 0.08).sqrt()
    visibility = (
        (-sight_distance_squared / (2.0 * 1.55 * 1.55)).exp()
        / (sight_distance_squared + 0.14)
    )
    signed_visibility = visibility * (
        1.20 * (1.0 - cargo_a) * trail_c
        - 0.30 * trail_a + 1.65 * (1.0 - cargo_a) * growing_food
        + 0.18 * angular_similarity
    )
    visibility_total = visibility.sum() + 1.0e-6
    steering_x = ((sight_x / sight_distance) * signed_visibility).sum()
    steering_x = column_x * 0.0 + steering_x / visibility_total
    steering_y = ((sight_y / sight_distance) * signed_visibility).sum()
    steering_y = column_y * 0.0 + steering_y / visibility_total
    separation_ab_x = entity_x - entity_b_x
    separation_ab_y = entity_y - entity_b_y
    separation_ac_x = entity_x - entity_c_x
    separation_ac_y = entity_y - entity_c_y
    separation_ab = (
        separation_ab_x * separation_ab_x
        + separation_ab_y * separation_ab_y + 0.18
    )
    separation_ac = (
        separation_ac_x * separation_ac_x
        + separation_ac_y * separation_ac_y + 0.18
    )
    nest_a_x = nest_x - entity_x
    nest_a_y = nest_y - entity_y
    nest_a_length = (nest_a_x * nest_a_x + nest_a_y * nest_a_y + 0.08).sqrt()
    acceleration_x = (
        2.35 * steering_x
        + 1.85 * cargo_a * nest_a_x / nest_a_length
        + 0.58 * (next_time * 1.71 + 0.20).cos()
        + 0.30 * separation_ab_x / separation_ab
        + 0.30 * separation_ac_x / separation_ac
        + 0.08 * (5.0 - entity_x) - 0.54 * entity_velocity_x
    )
    acceleration_y = (
        2.35 * steering_y
        + 1.85 * cargo_a * nest_a_y / nest_a_length
        + 0.58 * (next_time * 1.37 + 1.10).sin()
        + 0.30 * separation_ab_y / separation_ab
        + 0.30 * separation_ac_y / separation_ac
        + 0.08 * (3.5 - entity_y) - 0.54 * entity_velocity_y
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
    sight_b_distance = (sight_b_distance_squared + 0.08).sqrt()
    visibility_b = (
        (-sight_b_distance_squared / (2.0 * 1.55 * 1.55)).exp()
        / (sight_b_distance_squared + 0.14)
    )
    signed_visibility_b = visibility_b * (
        1.20 * (1.0 - cargo_b) * trail_a - 0.30 * trail_b
        + 1.65 * (1.0 - cargo_b) * growing_food
        + 0.18 * (region_y * preferred_x - region_x * preferred_y)
    )
    visibility_b_total = visibility_b.sum() + 1.0e-6
    steering_b_x = ((sight_b_x / sight_b_distance) * signed_visibility_b).sum()
    steering_b_x = column_x * 0.0 + steering_b_x / visibility_b_total
    steering_b_y = ((sight_b_y / sight_b_distance) * signed_visibility_b).sum()
    steering_b_y = column_y * 0.0 + steering_b_y / visibility_b_total
    separation_bc_x = entity_b_x - entity_c_x
    separation_bc_y = entity_b_y - entity_c_y
    separation_bc = (
        separation_bc_x * separation_bc_x
        + separation_bc_y * separation_bc_y + 0.18
    )
    nest_b_x = nest_x - entity_b_x
    nest_b_y = nest_y - entity_b_y
    nest_b_length = (nest_b_x * nest_b_x + nest_b_y * nest_b_y + 0.08).sqrt()
    acceleration_b_x = (
        2.35 * steering_b_x
        + 1.85 * cargo_b * nest_b_x / nest_b_length
        + 0.58 * (next_time * 1.63 + 2.30).cos()
        - 0.30 * separation_ab_x / separation_ab
        + 0.30 * separation_bc_x / separation_bc
        + 0.08 * (5.0 - entity_b_x) - 0.54 * entity_b_velocity_x
    )
    acceleration_b_y = (
        2.35 * steering_b_y
        + 1.85 * cargo_b * nest_b_y / nest_b_length
        + 0.58 * (next_time * 1.43 + 2.80).sin()
        - 0.30 * separation_ab_y / separation_ab
        + 0.30 * separation_bc_y / separation_bc
        + 0.08 * (3.5 - entity_b_y) - 0.54 * entity_b_velocity_y
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
    sight_c_distance = (sight_c_distance_squared + 0.08).sqrt()
    visibility_c = (
        (-sight_c_distance_squared / (2.0 * 1.55 * 1.55)).exp()
        / (sight_c_distance_squared + 0.14)
    )
    signed_visibility_c = visibility_c * (
        1.20 * (1.0 - cargo_c) * trail_b - 0.30 * trail_c
        + 1.65 * (1.0 - cargo_c) * growing_food
        - 0.18 * angular_similarity
    )
    visibility_c_total = visibility_c.sum() + 1.0e-6
    steering_c_x = ((sight_c_x / sight_c_distance) * signed_visibility_c).sum()
    steering_c_x = column_x * 0.0 + steering_c_x / visibility_c_total
    steering_c_y = ((sight_c_y / sight_c_distance) * signed_visibility_c).sum()
    steering_c_y = column_y * 0.0 + steering_c_y / visibility_c_total
    nest_c_x = nest_x - entity_c_x
    nest_c_y = nest_y - entity_c_y
    nest_c_length = (nest_c_x * nest_c_x + nest_c_y * nest_c_y + 0.08).sqrt()
    acceleration_c_x = (
        2.35 * steering_c_x
        + 1.85 * cargo_c * nest_c_x / nest_c_length
        + 0.58 * (next_time * 1.79 + 4.20).cos()
        - 0.30 * separation_ac_x / separation_ac
        - 0.30 * separation_bc_x / separation_bc
        + 0.08 * (5.0 - entity_c_x) - 0.54 * entity_c_velocity_x
    )
    acceleration_c_y = (
        2.35 * steering_c_y
        + 1.85 * cargo_c * nest_c_y / nest_c_length
        + 0.58 * (next_time * 1.31 + 5.00).sin()
        - 0.30 * separation_ac_y / separation_ac
        - 0.30 * separation_bc_y / separation_bc
        + 0.08 * (3.5 - entity_c_y) - 0.54 * entity_c_velocity_y
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
    pickup_a = (1.0 - cargo_a) * food_near_a.minimum(0.7) * 1.35
    pickup_b = (1.0 - cargo_b) * food_near_b.minimum(0.7) * 1.35
    pickup_c = (1.0 - cargo_c) * food_near_c.minimum(0.7) * 1.35

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

    drain_distance = (
        (column_x - 1.15) * (column_x - 1.15)
        + (column_y - 5.75) * (column_y - 5.75)
    )
    return_distance = (
        (column_x - 8.85) * (column_x - 8.85)
        + (column_y - 5.75) * (column_y - 5.75)
    )
    drain_mask = (-drain_distance / (2.0 * 0.48 * 0.48)).exp()
    return_mask = (-return_distance / (2.0 * 0.58 * 0.58)).exp()
    next_food_store = (
        (growing_food - food_consumption).maximum(0.0)
        * (1.0 - 0.10 * dt * drain_mask)
    ).minimum(1.0)
    delivered_food = dt * (delivery_a + delivery_b + delivery_c)
    next_nest_food = (nest_food + delivered_food).maximum(0.0)
    clean_release = filter_reservoir.maximum(0.0).minimum(0.42)
    target = (
        -0.42 * load - 0.22 * entity_interior * entity_interior
        + 0.16 * clean_release * return_mask
    )
    acceleration = (
        20.0 * (target - displacement) - 4.6 * displacement_velocity
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
    next_ink_red = (
        ink_red * (-0.095 * dt).exp() + 3.2 * dt * source_red * pulse_a
    ).minimum(1.0)
    next_ink_yellow = (
        ink_yellow * (-0.072 * dt).exp()
        + 2.2 * dt * source_yellow * pulse_a
    ).minimum(1.0)
    next_ink_green = (
        ink_green * (-0.095 * dt).exp() + 3.2 * dt * source_green * pulse_b
    ).minimum(1.0)
    next_ink_cyan = (
        ink_cyan * (-0.072 * dt).exp() + 2.2 * dt * source_cyan * pulse_b
    ).minimum(1.0)
    next_ink_blue = (
        ink_blue * (-0.095 * dt).exp() + 3.2 * dt * source_blue * pulse_c
    ).minimum(1.0)
    next_ink_magenta = (
        ink_magenta * (-0.072 * dt).exp()
        + 2.2 * dt * source_magenta * pulse_c
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
    next_ink_red = next_ink_red * (1.0 - drain_fraction)
    next_ink_yellow = next_ink_yellow * (1.0 - drain_fraction)
    next_ink_green = next_ink_green * (1.0 - drain_fraction)
    next_ink_cyan = next_ink_cyan * (1.0 - drain_fraction)
    next_ink_blue = next_ink_blue * (1.0 - drain_fraction)
    next_ink_magenta = next_ink_magenta * (1.0 - drain_fraction)
    clean_emission = 0.30 * dt * clean_release
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
    red = base_red * (1.0 - ink_alpha) + ink_color_red * ink_alpha
    green = base_green * (1.0 - ink_alpha) + ink_color_green * ink_alpha
    blue = base_blue * (1.0 - ink_alpha) + ink_color_blue * ink_alpha

    food_alpha = next_food_store.minimum(0.76)
    red = red * (1.0 - food_alpha) + 226.0 * food_alpha
    green = green * (1.0 - food_alpha) + 181.0 * food_alpha
    blue = blue * (1.0 - food_alpha) + 62.0 * food_alpha
    nest_field_distance = (
        (column_x - nest_x) * (column_x - nest_x)
        + (column_y - nest_y) * (column_y - nest_y)
    )
    nest_body = (-nest_field_distance / (2.0 * 0.34 * 0.34)).exp()
    nest_glow = next_nest_food.minimum(1.0)
    nest_red = 102.0 + 58.0 * nest_glow
    nest_green = 72.0 + 42.0 * nest_glow
    nest_blue = 48.0 + 24.0 * nest_glow
    red = red * (1.0 - nest_body) + nest_red * nest_body
    green = green * (1.0 - nest_body) + nest_green * nest_body
    blue = blue * (1.0 - nest_body) + nest_blue * nest_body

    clean_alpha = (return_mask * clean_release).minimum(0.60)
    red = red * (1.0 - clean_alpha) + 218.0 * clean_alpha
    green = green * (1.0 - clean_alpha) + 249.0 * clean_alpha
    blue = blue * (1.0 - clean_alpha) + 255.0 * clean_alpha
    drain_body = (-drain_distance / (2.0 * 0.23 * 0.23)).exp()
    return_body = (-return_distance / (2.0 * 0.23 * 0.23)).exp()
    red = red * (1.0 - drain_body) + 53.0 * drain_body
    green = green * (1.0 - drain_body) + 83.0 * drain_body
    blue = blue * (1.0 - drain_body) + 103.0 * drain_body
    red = red * (1.0 - return_body) + 205.0 * return_body
    green = green * (1.0 - return_body) + 245.0 * return_body
    blue = blue * (1.0 - return_body) + 252.0 * return_body
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


def columnar_multifluid_present(red, green, blue):
    """Python-authored display expressions lowered to a packed WebGL color."""

    display_red = red / 255.0
    display_green = green / 255.0
    display_blue = blue / 255.0
    return display_red, display_green, display_blue
