
TURING_PAGE = {
    "entrypoint": "columnar_multifluid_rgb_step",
    "title": "Managed Columnar Multifluid World",
    "slug": "managed-columnar-multifluid-world",
    "width": 384,
    "height": 268,
    "probe_size": 16,
    "feeds": {
        "column_x": 0.5,
        "column_y": 0.5,
        "rest_surface": 1.0,
        "displacement": 0.0,
        "displacement_velocity": 0.0,
        "managed_time": 0.0,
        "dt": 0.025
    },
    "feed_expressions": {
        "column_x": "(x + 0.5) * 10.0 / w",
        "column_y": "(y + 0.5) * 7.0 / h",
        "rest_surface": "1.15 + 3.1 * Math.exp(-16.0 * (((x + 0.5) / w - 0.62) ** 2 + ((y + 0.5) / h - 0.52) ** 2))",
        "displacement": "0.0",
        "displacement_velocity": "0.0",
        "managed_time": "0.0",
        "dt": "0.025"
    },
    "state_feedback": {
        "displacement": "next_displacement",
        "displacement_velocity": "next_velocity",
        "managed_time": "next_time"
    },
    "render_fps": 30.0,
    "autostart": True,
    "backend": "c",
    "remove_loops": True
}


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
