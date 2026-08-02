"""Small uploadable example for the local compiler-page publisher."""

TURING_PAGE = {
    "title": "Affine Field",
    "slug": "affine-field",
    "entrypoint": "render",
    "feeds": {"gain": {"values": [0.8, 0.8, 0.8, 0.8]}},
    "feed_expressions": {
        "unit_x": "x / Math.max(1, w - 1)",
        "unit_y": "y / Math.max(1, h - 1)",
    },
    "width": 96,
    "height": 64,
}


def render(unit_x, unit_y, gain):
    """Produce a scalar field which the generated page can render as pixels."""

    return (unit_x * gain + unit_y * (1.0 - gain)) * 255.0
