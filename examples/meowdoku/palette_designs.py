"""The color game's designs: a named CSS color with a bevel and a shadow, the features the network reads, and the CSS a
page wears. Ticket 7f79d9a4, case 9f272724.

A design is drawn at random: one of the 148 named CSS colors (CSS Color 4), a bevel (depth, light angle) and a shadow
(x, y, blur, spread, darkness). Its feature vector is FEATURES floats in [0, 1]: the color in OKLab (perceptual, so the
network sees distances people see), the bevel depth with the sine and cosine of its light angle, and the five shadow
numbers. ``site_css`` turns a chosen design into what palette-theme.js sets; the accent and secondary are moved in
OKLab lightness, keeping the hue, until they meet the contrast rule against the site's dark background.
"""
from __future__ import annotations

import math
import random

NAMED = {
    "aliceblue": "#f0f8ff", "antiquewhite": "#faebd7", "aqua": "#00ffff", "aquamarine": "#7fffd4", "azure": "#f0ffff",
    "beige": "#f5f5dc", "bisque": "#ffe4c4", "black": "#000000", "blanchedalmond": "#ffebcd", "blue": "#0000ff",
    "blueviolet": "#8a2be2", "brown": "#a52a2a", "burlywood": "#deb887", "cadetblue": "#5f9ea0", "chartreuse": "#7fff00",
    "chocolate": "#d2691e", "coral": "#ff7f50", "cornflowerblue": "#6495ed", "cornsilk": "#fff8dc", "crimson": "#dc143c",
    "cyan": "#00ffff", "darkblue": "#00008b", "darkcyan": "#008b8b", "darkgoldenrod": "#b8860b", "darkgray": "#a9a9a9",
    "darkgreen": "#006400", "darkgrey": "#a9a9a9", "darkkhaki": "#bdb76b", "darkmagenta": "#8b008b",
    "darkolivegreen": "#556b2f", "darkorange": "#ff8c00", "darkorchid": "#9932cc", "darkred": "#8b0000",
    "darksalmon": "#e9967a", "darkseagreen": "#8fbc8f", "darkslateblue": "#483d8b", "darkslategray": "#2f4f4f",
    "darkslategrey": "#2f4f4f", "darkturquoise": "#00ced1", "darkviolet": "#9400d3", "deeppink": "#ff1493",
    "deepskyblue": "#00bfff", "dimgray": "#696969", "dimgrey": "#696969", "dodgerblue": "#1e90ff",
    "firebrick": "#b22222", "floralwhite": "#fffaf0", "forestgreen": "#228b22", "fuchsia": "#ff00ff",
    "gainsboro": "#dcdcdc", "ghostwhite": "#f8f8ff", "gold": "#ffd700", "goldenrod": "#daa520", "gray": "#808080",
    "green": "#008000", "greenyellow": "#adff2f", "grey": "#808080", "honeydew": "#f0fff0", "hotpink": "#ff69b4",
    "indianred": "#cd5c5c", "indigo": "#4b0082", "ivory": "#fffff0", "khaki": "#f0e68c", "lavender": "#e6e6fa",
    "lavenderblush": "#fff0f5", "lawngreen": "#7cfc00", "lemonchiffon": "#fffacd", "lightblue": "#add8e6",
    "lightcoral": "#f08080", "lightcyan": "#e0ffff", "lightgoldenrodyellow": "#fafad2", "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90", "lightgrey": "#d3d3d3", "lightpink": "#ffb6c1", "lightsalmon": "#ffa07a",
    "lightseagreen": "#20b2aa", "lightskyblue": "#87cefa", "lightslategray": "#778899", "lightslategrey": "#778899",
    "lightsteelblue": "#b0c4de", "lightyellow": "#ffffe0", "lime": "#00ff00", "limegreen": "#32cd32", "linen": "#faf0e6",
    "magenta": "#ff00ff", "maroon": "#800000", "mediumaquamarine": "#66cdaa", "mediumblue": "#0000cd",
    "mediumorchid": "#ba55d3", "mediumpurple": "#9370db", "mediumseagreen": "#3cb371", "mediumslateblue": "#7b68ee",
    "mediumspringgreen": "#00fa9a", "mediumturquoise": "#48d1cc", "mediumvioletred": "#c71585",
    "midnightblue": "#191970", "mintcream": "#f5fffa", "mistyrose": "#ffe4e1", "moccasin": "#ffe4b5",
    "navajowhite": "#ffdead", "navy": "#000080", "oldlace": "#fdf5e6", "olive": "#808000", "olivedrab": "#6b8e23",
    "orange": "#ffa500", "orangered": "#ff4500", "orchid": "#da70d6", "palegoldenrod": "#eee8aa", "palegreen": "#98fb98",
    "paleturquoise": "#afeeee", "palevioletred": "#db7093", "papayawhip": "#ffefd5", "peachpuff": "#ffdab9",
    "peru": "#cd853f", "pink": "#ffc0cb", "plum": "#dda0dd", "powderblue": "#b0e0e6", "purple": "#800080",
    "rebeccapurple": "#663399", "red": "#ff0000", "rosybrown": "#bc8f8f", "royalblue": "#4169e1",
    "saddlebrown": "#8b4513", "salmon": "#fa8072", "sandybrown": "#f4a460", "seagreen": "#2e8b57",
    "seashell": "#fff5ee", "sienna": "#a0522d", "silver": "#c0c0c0", "skyblue": "#87ceeb", "slateblue": "#6a5acd",
    "slategray": "#708090", "slategrey": "#708090", "snow": "#fffafa", "springgreen": "#00ff7f", "steelblue": "#4682b4",
    "tan": "#d2b48c", "teal": "#008080", "thistle": "#d8bfd8", "tomato": "#ff6347", "turquoise": "#40e0d0",
    "violet": "#ee82ee", "wheat": "#f5deb3", "white": "#ffffff", "whitesmoke": "#f5f5f5", "yellow": "#ffff00",
    "yellowgreen": "#9acd32",
}
NAMES = tuple(sorted(NAMED))
FEATURES = 11
BACKGROUND = "#121822"          # the site's dark page color (index.html, pages.css): what accents must stand out on
ACCENT_CONTRAST, SECONDARY_CONTRAST = 4.5, 3.0     # WCAG AA for text; 3:1 for the secondary (large marks, borders)
BEVEL_MAX, SHADOW_X, SHADOW_Y, BLUR_MAX, SPREAD, ALPHA_MAX = 6.0, 8.0, 14.0, 24.0, 4.0, 0.7


def _hex_rgb(value):
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def _linear(channel):
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _gamma(channel):
    channel = min(1.0, max(0.0, channel))
    return 12.92 * channel if channel <= 0.0031308 else 1.055 * channel ** (1 / 2.4) - 0.055


def oklab(value):
    """sRGB hex -> OKLab (L, a, b), Björn Ottosson's matrices."""
    r, g, b = (_linear(c) for c in _hex_rgb(value))
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _oklab_linear(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def _hex(linear_rgb):
    return "#" + "".join(f"{round(255 * _gamma(c)):02x}" for c in linear_rgb)


def luminance(value):
    r, g, b = (_linear(c) for c in _hex_rgb(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(one, two):
    high, low = sorted((luminance(one), luminance(two)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def readable(value, ratio, background=BACKGROUND):
    """The color moved in OKLab lightness, hue kept, until it has ``ratio`` contrast on ``background``; chroma is
    reduced only as far as the sRGB gamut needs. A color that already has it is returned as it was."""
    if contrast(value, background) >= ratio:
        return value
    L, a, b = oklab(value)
    for step in range(1, 101):
        lightness = min(1.0, L + (1.0 - L) * step / 100)
        for chroma in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
            linear = _oklab_linear((lightness, a * chroma, b * chroma))
            if all(-1e-4 <= c <= 1 + 1e-4 for c in linear):
                candidate = _hex(linear)
                if contrast(candidate, background) >= ratio:
                    return candidate
                break
    return "#ffffff"


def random_round(rng: random.Random, count: int) -> list:
    """``count`` random designs of different colors: no two squares of one
    round share a named color (or, for the CSS aliases such as ``aqua`` and
    ``cyan``, the same hex)."""
    colors = sorted({NAMED[name] for name in NAMES})
    chosen = rng.sample(colors, count)
    names = {color: sorted(name for name in NAMES if NAMED[name] == color) for color in chosen}
    return [random_design(rng, name=rng.choice(names[color])) for color in chosen]


def random_design(rng: random.Random, name: str | None = None) -> dict:
    name = rng.choice(NAMES) if name is None else name
    return {"name": name, "color": NAMED[name],
            "bevel": round(rng.uniform(0.0, BEVEL_MAX), 2), "light": round(rng.uniform(0.0, 2 * math.pi), 3),
            "shadow_x": round(rng.uniform(-SHADOW_X, SHADOW_X), 2), "shadow_y": round(rng.uniform(0.0, SHADOW_Y), 2),
            "blur": round(rng.uniform(0.0, BLUR_MAX), 2), "spread": round(rng.uniform(-SPREAD, SPREAD), 2),
            "alpha": round(rng.uniform(0.0, ALPHA_MAX), 3)}


def valid(design) -> bool:
    try:
        return (design["name"] in NAMED and design["color"] == NAMED[design["name"]]
                and 0 <= float(design["bevel"]) <= BEVEL_MAX and 0 <= float(design["light"]) <= 2 * math.pi + 1e-6
                and abs(float(design["shadow_x"])) <= SHADOW_X and 0 <= float(design["shadow_y"]) <= SHADOW_Y
                and 0 <= float(design["blur"]) <= BLUR_MAX and abs(float(design["spread"])) <= SPREAD
                and 0 <= float(design["alpha"]) <= ALPHA_MAX)
    except (KeyError, TypeError, ValueError):
        return False


def features(design) -> list:
    """FEATURES floats in [0, 1] (OKLab a and b span about +-0.4 over the sRGB gamut)."""
    L, a, b = oklab(design["color"])
    return [L, (a + 0.4) / 0.8, (b + 0.4) / 0.8,
            design["bevel"] / BEVEL_MAX, (math.sin(design["light"]) + 1) / 2, (math.cos(design["light"]) + 1) / 2,
            (design["shadow_x"] + SHADOW_X) / (2 * SHADOW_X), design["shadow_y"] / SHADOW_Y, design["blur"] / BLUR_MAX,
            (design["spread"] + SPREAD) / (2 * SPREAD), design["alpha"] / ALPHA_MAX]


def bevel_css(design) -> str:
    depth = design["bevel"]
    if depth <= 0.05:
        return "inset 0 0 0 0 transparent"
    dx, dy = -math.cos(design["light"]) * depth, -math.sin(design["light"]) * depth
    return (f"inset {dx:.1f}px {dy:.1f}px 0 rgba(255,255,255,0.28), "
            f"inset {-dx:.1f}px {-dy:.1f}px 0 rgba(0,0,0,0.38)")


def shadow_css(design) -> str:
    return (f"{design['shadow_x']:.1f}px {design['shadow_y']:.1f}px {design['blur']:.1f}px {design['spread']:.1f}px "
            f"rgba(0,0,0,{design['alpha']:.2f})")


def square_css(design) -> dict:
    """What one square of the field wears: its own color, bevel and shadow."""
    return {"background": design["color"], "boxShadow": f"{bevel_css(design)}, {shadow_css(design)}"}


def site_css(primary, secondary) -> dict:
    """What palette-theme.js sets for a design: readable accent and secondary, the bevel and shadow of the primary."""
    return {"color": readable(primary["color"], ACCENT_CONTRAST),
            "secondary": readable(secondary["color"], SECONDARY_CONTRAST),
            "bevel": bevel_css(primary), "shadow": shadow_css(primary)}
