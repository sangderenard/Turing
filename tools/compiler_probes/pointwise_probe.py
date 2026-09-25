"""A minimal, well-formed page: per-pixel, state-fed, naming red/green/blue.

Nothing here is stencil-shaped. It exists to answer one question -- does the
site generator publish a page again after the naming, flattening and
boundary-dtype repairs -- without a malformed program muddying the answer.
"""

TURING_PAGE = {
    "entrypoint": "step",
    "title": "Pointwise probe",
    "slug": "pointwise-probe",
    "width": 32,
    "height": 32,
    "feeds": {"phase": 0.0, "rate": 0.0, "dt": 0.05},
    "feed_expressions": {
        "phase": "((x + 0.5) / w) * 6.283185307179586",
        "rate": "0.4 + 0.6 * ((y + 0.5) / h)",
        "dt": "0.05",
    },
    "state_feedback": {"phase": "next_phase"},
    "render_fps": 30.0,
    "autostart": True,
    "backend": "c",
    "shader_configuration": {
        "channels": ["red", "green", "blue"],
        "channel_scale": 255.0,
    },
}


def step(phase, rate, dt):
    """One turn of a per-pixel oscillator, painted as hue."""

    next_phase = phase + dt * rate
    red = 0.5 + 0.5 * next_phase.cos()
    green = 0.5 + 0.5 * (next_phase - 2.094395102393195).cos()
    blue = 0.5 + 0.5 * (next_phase - 4.18879020478639).cos()
    return next_phase, red, green, blue
