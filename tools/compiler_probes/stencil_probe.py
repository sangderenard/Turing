"""Does the site generator's program model accept a stencil?

Written to be compiled by the generator, not to be pretty. The question is
narrow: can the entrypoint hold the field as a tensor and index its
neighbours the ordinary way -- ``theta[up]`` -- rather than needing some
per-pixel neighbour concept invented for the occasion.
"""

TURING_PAGE = {
    "entrypoint": "step",
    "title": "Stencil probe",
    "slug": "stencil-probe",
    "width": 16,
    "height": 16,
    # EVERY parameter is a declared feed. Leaving three undeclared made
    # the compiled inputs come back as feed0/feed1 with no names, and the
    # state feedback then had nothing called "theta" to bind to.
    "feeds": {"theta": 0.0, "dt": 0.05, "wide": 16, "high": 16,
              "index": 0},
    "feed_expressions": {
        "theta": "Math.sin((x + 0.5) * 0.4) * Math.cos((y + 0.5) * 0.4)",
        "dt": "0.05",
        "wide": "w",
        "high": "h",
        "index": "y * w + x",
    },
    "state_feedback": {"theta": "next_theta"},
    "render_fps": 30.0,
    "autostart": True,
    "backend": "c",
}


def step(theta, dt, wide, high, index):
    """One diffusion tick, reading four neighbours by ordinary indexing."""

    row = index // wide
    column = index - row * wide
    up = ((row + high - 1) % high) * wide + column
    down = ((row + 1) % high) * wide + column
    left = row * wide + (column + wide - 1) % wide
    right = row * wide + (column + 1) % wide

    here = theta[index]
    pull = theta[up] + theta[down] + theta[left] + theta[right] - 4.0 * here
    next_theta = here + dt * pull

    red = 0.5 + 0.5 * next_theta
    green = 0.5 - 0.5 * next_theta
    blue = 0.5 + 0.0 * next_theta
    return next_theta, red, green, blue
