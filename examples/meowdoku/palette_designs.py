"""The color game's designs: a palette of named CSS colors and a token for every architectural element of a page, the
features the network reads, and the CSS a page wears.

A design is a palette of 1 to 6 colors (``SIZE_WEIGHTS``: a normal curve centred on 4, as a decorator's palette is)
drawn from the 148 named CSS colors by a harmony scheme (``SCHEMES``: monochrome, analogous, complementary,
split-complementary, triadic, tetradic, or free), plus a token for each element of ``ELEMENTS`` drawn around a style
family (``STYLES``: flat, material, beveled, brutalist, soft, glass, editorial, or free). An element's token says which
palette color it takes (``slot``), where its lightness sits in the element's band (``tone``), how much of that color's
chroma it keeps (``chroma``), and, for boxes, its corner radius, border width, bevel and shadow. The light angle is one
per design, so every bevel and shadow on a page is lit from the same side.

The features (``features``, ``FEATURES`` floats, about 0..1) are, per element, the slot color in OKLab, the token's
tone and chroma, the element's resolved (dark) color in OKLab and its geometry; then the palette's harmony scores
(``harmony``: affinity for monochrome, analogous, complementary, triadic, split and tetradic arrangements, hue spread
and entropy, neutrals, lightness and chroma spread, warmth, distinctness) and the page's structure (contrasts and
lightness steps between page, surface, text, links and buttons). The theory measures; the network learns what is good.

Readability is not learned: when a design is worn, text, links, muted text and the focus ring are moved in OKLab
lightness (hue kept) until they meet their contrast on the surface behind them (``CONTRAST``).

Designs stored before palettes (one named color with a bevel and shadow, worn with a second design as its secondary)
are read by ``upgrade`` as palettes of those colors with the default tokens.
"""
from __future__ import annotations

import itertools
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
BACKGROUND = "#121822"          # the site's dark page color before designs: kept for the legacy accent rule

# -- the palette ------------------------------------------------------------------------------------------------------
SIZES = (1, 2, 3, 4, 5, 6)
SIZE_SIGMA = 1.2
SIZE_WEIGHTS = tuple(math.exp(-((n - 4) ** 2) / (2 * SIZE_SIGMA ** 2)) for n in SIZES)
NEUTRAL_CHROMA = 0.04            # OKLCH chroma below which a color reads as a neutral (white, grays, black, linen...)
NEUTRAL_SHARE = 0.3              # chance a palette color after the first is drawn as a neutral
# scheme -> hue offsets in degrees from the base hue; colors beyond the offsets cycle through them.
SCHEMES = {
    "monochrome": (0,),
    "analogous": (0, 30, -30, 60, -60),
    "complementary": (0, 180),
    "split": (0, 150, 210),
    "triadic": (0, 120, 240),
    "tetradic": (0, 90, 180, 270),
    "free": None,
}
SCHEME_WEIGHTS = {"monochrome": 1, "analogous": 1, "complementary": 1, "split": 1, "triadic": 1, "tetradic": 1,
                  "free": 2}

# -- the page's elements ----------------------------------------------------------------------------------------------
# element -> (role it takes by default, is it a box with geometry, dark lightness band, light lightness band)
ELEMENTS = {
    "page":      ("dominant", False, (0.10, 0.24), (0.95, 0.995)),
    "surface":   ("dominant", True,  (0.16, 0.32), (0.97, 1.0)),
    "raised":    ("dominant", True,  (0.24, 0.40), (0.89, 0.96)),
    "nav":       ("dominant", True,  (0.12, 0.30), (0.90, 0.99)),
    "header":    ("dominant", True,  (0.12, 0.36), (0.88, 0.99)),
    "input":     ("dominant", True,  (0.14, 0.28), (0.97, 1.0)),
    "line":      ("dominant", False, (0.30, 0.50), (0.74, 0.88)),
    "text":      ("dominant", False, (0.88, 0.97), (0.14, 0.26)),
    "muted":     ("dominant", False, (0.64, 0.80), (0.40, 0.56)),
    "link":      ("accent", False, (0.70, 0.90), (0.34, 0.54)),
    "button":    ("accent", True,  (0.45, 0.78), (0.42, 0.70)),
    "button2":   ("secondary", True,  (0.24, 0.46), (0.84, 0.94)),
    "badge":     ("secondary", True,  (0.30, 0.56), (0.84, 0.94)),
    "secondary": ("secondary", False, (0.55, 0.85), (0.40, 0.66)),
    "focus":     ("accent", False, (0.62, 0.90), (0.40, 0.64)),
    "stripe":    ("accent", False, (0.40, 0.80), (0.44, 0.76)),
}
ELEMENT_NAMES = tuple(ELEMENTS)
BOXES = tuple(name for name, spec in ELEMENTS.items() if spec[1])
# Readability, not learned: element -> (the element behind it, contrast it must have there).
CONTRAST = {"text": ("surface", 4.5), "muted": ("surface", 3.0), "link": ("surface", 4.5), "focus": ("page", 3.0),
            "secondary": ("page", 3.0)}
RADIUS_MAX, BORDER_MAX, BEVEL_MAX, DEPTH_MAX, BLUR_MAX, ALPHA_MAX = 24.0, 3.0, 6.0, 16.0, 32.0, 0.8
REASSIGN = 0.15                  # chance an element takes a palette color other than its role's

# Style families: attribute -> (low, high) for boxes, and the chroma ranges of surfaces, text and accents.
STYLES = {
    "flat":      {"radius": (2, 8), "border": (0, 1), "bevel": (0, 0), "depth": (0, 0), "blur": (0, 0),
                  "alpha": (0, 0), "surface_chroma": (0.0, 0.5), "text_chroma": (0.0, 0.2), "accent_chroma": (0.7, 1.2)},
    "material":  {"radius": (4, 10), "border": (0, 0), "bevel": (0, 0), "depth": (2, 8), "blur": (4, 16),
                  "alpha": (0.2, 0.45), "surface_chroma": (0.0, 0.3), "text_chroma": (0.0, 0.1), "accent_chroma": (0.8, 1.2)},
    "beveled":   {"radius": (3, 10), "border": (1, 1), "bevel": (2, 5), "depth": (3, 8), "blur": (6, 12),
                  "alpha": (0.35, 0.6), "surface_chroma": (0.1, 0.6), "text_chroma": (0.0, 0.2), "accent_chroma": (0.6, 1.1)},
    "brutalist": {"radius": (0, 0), "border": (2, 3), "bevel": (0, 0), "depth": (4, 8), "blur": (0, 0),
                  "alpha": (0.6, 0.8), "surface_chroma": (0.2, 1.0), "text_chroma": (0.0, 0.3), "accent_chroma": (0.9, 1.2)},
    "soft":      {"radius": (12, 24), "border": (0, 0), "bevel": (1, 3), "depth": (4, 12), "blur": (16, 32),
                  "alpha": (0.15, 0.35), "surface_chroma": (0.0, 0.25), "text_chroma": (0.0, 0.15), "accent_chroma": (0.4, 0.8)},
    "glass":     {"radius": (12, 20), "border": (1, 1), "bevel": (1, 2), "depth": (6, 14), "blur": (20, 32),
                  "alpha": (0.25, 0.45), "surface_chroma": (0.2, 0.6), "text_chroma": (0.0, 0.1), "accent_chroma": (0.6, 1.0)},
    "editorial": {"radius": (0, 2), "border": (1, 1), "bevel": (0, 0), "depth": (0, 0), "blur": (0, 0),
                  "alpha": (0, 0), "surface_chroma": (0.0, 0.15), "text_chroma": (0.0, 0.1), "accent_chroma": (0.5, 0.9)},
    "free":      {"radius": (0, RADIUS_MAX), "border": (0, BORDER_MAX), "bevel": (0, BEVEL_MAX), "depth": (0, DEPTH_MAX),
                  "blur": (0, BLUR_MAX), "alpha": (0, ALPHA_MAX), "surface_chroma": (0.0, 1.2),
                  "text_chroma": (0.0, 1.2), "accent_chroma": (0.0, 1.2)},
}
STYLE_WEIGHTS = {"flat": 1, "material": 1, "beveled": 1, "brutalist": 1, "soft": 1, "glass": 1, "editorial": 1,
                 "free": 1}
GEOMETRY = ("radius", "border", "bevel", "depth", "blur", "alpha")
_GEOMETRY_MAX = {"radius": RADIUS_MAX, "border": BORDER_MAX, "bevel": BEVEL_MAX, "depth": DEPTH_MAX,
                 "blur": BLUR_MAX, "alpha": ALPHA_MAX}
DEFAULT_TOKEN = {"slot": 0, "tone": 0.5, "chroma": 0.3, "radius": 8.0, "border": 1.0, "bevel": 0.0, "depth": 0.0,
                 "blur": 0.0, "alpha": 0.0}

PER_ELEMENT = 17                 # slot color OKLab 3, tone, chroma, resolved OKLab 3, geometry 6, light sin/cos, box flag
HARMONY = 16
STRUCTURE = 6
FEATURES = len(ELEMENTS) * PER_ELEMENT + HARMONY + STRUCTURE


# -- color arithmetic -------------------------------------------------------------------------------------------------
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


def oklch(value):
    L, a, b = oklab(value)
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360.0


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


def _in_gamut(linear):
    return all(-1e-4 <= c <= 1 + 1e-4 for c in linear)


def from_oklch(L, C, h):
    """OKLCH -> sRGB hex; chroma is reduced only as far as the sRGB gamut needs (hue and lightness kept)."""
    L = min(1.0, max(0.0, L))
    a, b = math.cos(math.radians(h)), math.sin(math.radians(h))
    low, high = 0.0, max(0.0, C)
    if _in_gamut(_oklab_linear((L, a * high, b * high))):
        return _hex(_oklab_linear((L, a * high, b * high)))
    for _ in range(24):
        mid = (low + high) / 2
        if _in_gamut(_oklab_linear((L, a * mid, b * mid))):
            low = mid
        else:
            high = mid
    return _hex(_oklab_linear((L, a * low, b * low)))


def luminance(value):
    r, g, b = (_linear(c) for c in _hex_rgb(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(one, two):
    high, low = sorted((luminance(one), luminance(two)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def readable(value, ratio, background=BACKGROUND):
    """The color moved in OKLab lightness, hue kept, away from ``background`` until it has ``ratio`` contrast on it;
    chroma is reduced only as far as the sRGB gamut needs. A color that already has it is returned as it was."""
    if contrast(value, background) >= ratio:
        return value
    L, C, h = oklch(value)
    darker = oklab(background)[0] > 0.6
    for step in range(1, 101):
        lightness = L - L * step / 100 if darker else L + (1.0 - L) * step / 100
        candidate = from_oklch(lightness, C, h)
        if contrast(candidate, background) >= ratio:
            return candidate
    return "#000000" if darker else "#ffffff"


# -- drawing designs --------------------------------------------------------------------------------------------------
_COLORS = sorted({NAMED[name] for name in NAMES})
_NAME_OF = {color: sorted(name for name in NAMES if NAMED[name] == color) for color in _COLORS}
_LCH = {color: oklch(color) for color in _COLORS}
_CHROMATIC = [color for color in _COLORS if _LCH[color][1] >= NEUTRAL_CHROMA]
_NEUTRALS = [color for color in _COLORS if _LCH[color][1] < NEUTRAL_CHROMA]


def _hue_gap(one, two):
    gap = abs(one - two) % 360.0
    return min(gap, 360.0 - gap)


def _weighted(rng, weights):
    names = list(weights)
    return rng.choices(names, weights=[weights[name] for name in names])[0]


def random_palette(rng: random.Random, size: int | None = None, scheme: str | None = None) -> tuple[list, str]:
    """``size`` distinct named colors arranged by ``scheme`` (both drawn when None); returns (colors, scheme)."""
    size = rng.choices(SIZES, weights=SIZE_WEIGHTS)[0] if size is None else size
    scheme = _weighted(rng, SCHEME_WEIGHTS) if scheme is None else scheme
    base = rng.choice(_CHROMATIC)
    chosen = [base]
    offsets = SCHEMES[scheme]
    for index in range(1, size):
        pool = [color for color in _COLORS if color not in chosen]
        if offsets is None:
            chosen.append(rng.choice(pool))
            continue
        if rng.random() < NEUTRAL_SHARE:
            neutrals = [color for color in _NEUTRALS if color not in chosen]
            if neutrals:
                chosen.append(rng.choice(neutrals))
                continue
        target = (_LCH[base][2] + offsets[index % len(offsets)]) % 360.0
        for tolerance in (15.0, 25.0, 40.0, 180.0):
            near = [color for color in pool if color in _CHROMATIC and _hue_gap(_LCH[color][2], target) <= tolerance]
            if near:
                chosen.append(rng.choice(near))
                break
    return [{"name": rng.choice(_NAME_OF[color]), "color": color} for color in chosen], scheme


def roles(colors) -> dict:
    """The palette's decorating roles: the most neutral color dominates (walls, the page's surfaces), the most
    chromatic is the accent, the next most chromatic the secondary."""
    by_chroma = sorted(range(len(colors)), key=lambda i: _LCH_of(colors[i]["color"])[1])
    dominant, accent = by_chroma[0], by_chroma[-1]
    rest = [i for i in reversed(by_chroma) if i not in (dominant, accent)]
    return {"dominant": dominant, "accent": accent, "secondary": rest[0] if rest else accent}


def _LCH_of(color):
    return _LCH.get(color) or oklch(color)


def random_design(rng: random.Random, *, size: int | None = None, scheme: str | None = None,
                  style: str | None = None) -> dict:
    colors, scheme = random_palette(rng, size, scheme)
    style = _weighted(rng, STYLE_WEIGHTS) if style is None else style
    ranges = STYLES[style]
    assigned = roles(colors)
    elements = {}
    for name, (role, box, _dark, _light) in ELEMENTS.items():
        slot = assigned[role]
        if len(colors) > 1 and rng.random() < REASSIGN:
            slot = rng.randrange(len(colors))
        kind = ("accent_chroma" if role != "dominant" else
                "text_chroma" if name in ("text", "muted", "line") else "surface_chroma")
        token = {"slot": slot, "tone": round(rng.random(), 3), "chroma": round(rng.uniform(*ranges[kind]), 3)}
        for attribute in GEOMETRY:
            low, high = ranges[attribute]
            token[attribute] = round(rng.uniform(low, high), 2) if box else 0.0
        elements[name] = token
    return {"version": 2, "style": style, "scheme": scheme, "colors": colors,
            "light": round(rng.uniform(0.0, 2 * math.pi), 3), "elements": elements}


def random_round(rng: random.Random, count: int) -> list:
    """``count`` random designs whose palettes start with different colors."""
    designs, firsts = [], set()
    while len(designs) < count:
        design = random_design(rng)
        if design["colors"][0]["color"] not in firsts:
            firsts.add(design["colors"][0]["color"])
            designs.append(design)
    return designs


def upgrade(design, secondary=None) -> dict:
    """A design of any stored version as a palette design. A design from before palettes (one named color with its
    bevel and shadow, worn beside a second design) becomes the palette of those colors, its bevel and shadow on the
    boxes, and the default tokens elsewhere."""
    if design.get("version") == 2:
        return design
    colors = [{"name": design["name"], "color": design["color"]}]
    if secondary and secondary.get("color") and secondary["color"] != design["color"]:
        colors.append({"name": secondary["name"], "color": secondary["color"]})
    assigned = roles(colors)
    elements = {}
    for name, (role, box, _dark, _light) in ELEMENTS.items():
        token = dict(DEFAULT_TOKEN, slot=assigned[role], chroma=0.3 if role == "dominant" else 1.0)
        if box:
            token.update(bevel=min(BEVEL_MAX, float(design.get("bevel", 0.0))),
                         depth=min(DEPTH_MAX, float(design.get("shadow_y", 0.0))),
                         blur=min(BLUR_MAX, float(design.get("blur", 0.0))),
                         alpha=min(ALPHA_MAX, float(design.get("alpha", 0.0))))
        else:
            token.update({attribute: 0.0 for attribute in GEOMETRY})
        elements[name] = token
    return {"version": 2, "style": "legacy", "scheme": "legacy", "colors": colors,
            "light": float(design.get("light", 0.0)), "elements": elements}


def valid(design) -> bool:
    try:
        colors = design["colors"]
        if design.get("version") != 2 or not 1 <= len(colors) <= 6:
            return False
        if any(NAMED.get(c["name"]) != c["color"] for c in colors) or len({c["color"] for c in colors}) != len(colors):
            return False
        if not 0 <= float(design["light"]) <= 2 * math.pi + 1e-6 or set(design["elements"]) != set(ELEMENTS):
            return False
        for name, token in design["elements"].items():
            if type(token["slot"]) is not int or not 0 <= token["slot"] < len(colors):
                return False
            if not 0 <= float(token["tone"]) <= 1 or not 0 <= float(token["chroma"]) <= 1.2:
                return False
            if any(not 0 <= float(token[a]) <= _GEOMETRY_MAX[a] for a in GEOMETRY):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


# -- resolving a design into colors -----------------------------------------------------------------------------------
def _resolve(design, scheme: str) -> dict:
    """Element -> hex for ``scheme`` 'dark' or 'light': the slot color's hue, its chroma scaled by the token, its
    lightness placed in the element's band by the token's tone; then the readability rules."""
    design = upgrade(design)
    colors = design["colors"]
    resolved = {}
    for name, (_role, _box, dark, light) in ELEMENTS.items():
        token = design["elements"][name]
        L, C, h = _LCH_of(colors[token["slot"]]["color"])
        low, high = dark if scheme == "dark" else light
        resolved[name] = from_oklch(low + (high - low) * float(token["tone"]), C * float(token["chroma"]), h)
    for name, (behind, ratio) in CONTRAST.items():
        resolved[name] = readable(resolved[name], ratio, resolved[behind])
    return resolved


def _on(fill):
    """Text on a fill: black or white, whichever has more contrast."""
    return "#000000" if contrast("#000000", fill) >= contrast("#ffffff", fill) else "#ffffff"


def _box_shadow(design, name) -> str:
    token = upgrade(design)["elements"][name]
    light = float(upgrade(design)["light"])
    dx, dy = -math.cos(light), -math.sin(light)
    parts = []
    bevel = float(token["bevel"])
    if bevel > 0.05:
        parts += [f"inset {dx * bevel:.1f}px {dy * bevel:.1f}px 0 rgba(255,255,255,0.28)",
                  f"inset {-dx * bevel:.1f}px {-dy * bevel:.1f}px 0 rgba(0,0,0,0.38)"]
    depth, blur, alpha = float(token["depth"]), float(token["blur"]), float(token["alpha"])
    if alpha > 0.01 and (depth > 0.05 or blur > 0.05):
        parts.append(f"{-dx * depth * 0.5:.1f}px {abs(dy) * depth + depth * 0.5:.1f}px {blur:.1f}px 0 "
                     f"rgba(0,0,0,{alpha:.2f})")
    return ", ".join(parts) if parts else "0 0 0 0 transparent"


# CSS custom property each element's color is worn as (pages.css and palette-theme.js read these).
COLOR_VARIABLES = {"page": "--palette-bg", "surface": "--palette-surface", "raised": "--palette-raised",
                   "line": "--palette-line", "text": "--palette-fg", "muted": "--palette-muted",
                   "link": "--palette-ink", "button": "--palette-accent", "secondary": "--palette-secondary",
                   "nav": "--palette-nav-bg", "header": "--palette-header-bg", "input": "--palette-input-bg",
                   "button2": "--palette-button2-bg", "badge": "--palette-badge-bg", "focus": "--palette-focus",
                   "stripe": "--palette-stripe"}


def wear(design, secondary=None) -> dict:
    """What palette-theme.js sets for a design: every element's color as light-dark(light, dark), the text color on
    each filled box, and each box's radius, border width and box-shadow. ``color``, ``secondary``, ``bevel`` and
    ``shadow`` are the dark accent, secondary and card bevel/shadow, for pages that read only those."""
    design = upgrade(design, secondary)
    dark, light = _resolve(design, "dark"), _resolve(design, "light")
    variables = {variable: f"light-dark({light[name]}, {dark[name]})" for name, variable in COLOR_VARIABLES.items()}
    variables["--palette-on-accent"] = f"light-dark({_on(light['button'])}, {_on(dark['button'])})"
    for name in BOXES:
        token = design["elements"][name]
        variables[f"--palette-{name}-radius"] = f"{float(token['radius']):.1f}px"
        variables[f"--palette-{name}-border"] = f"{float(token['border']):.1f}px"
        variables[f"--palette-{name}-shadow"] = _box_shadow(design, name)
    card = _box_shadow(design, "surface")
    return {"color": dark["button"], "secondary": dark["secondary"],
            "bevel": card.split(", ")[0] if "inset" in card else "inset 0 0 0 0 transparent",
            "shadow": card, "variables": variables}


def preview(design) -> dict:
    """The dark colors and geometry of a design, for the color game's mockups (drawn on the dark page)."""
    design = upgrade(design)
    dark = _resolve(design, "dark")
    view = {name: dark[name] for name in ELEMENTS}
    view["on_button"] = _on(dark["button"])
    for name in BOXES:
        token = design["elements"][name]
        view[f"{name}_radius"] = float(token["radius"])
        view[f"{name}_border"] = float(token["border"])
        view[f"{name}_shadow"] = _box_shadow(design, name)
    return view


# -- features ---------------------------------------------------------------------------------------------------------
def _lab_unit(hex_color):
    L, a, b = oklab(hex_color)
    return [L, (a + 0.4) / 0.8, (b + 0.4) / 0.8]


def _closeness(gaps, targets, width=20.0):
    """1 when every gap equals its target (in degrees), falling off as a Gaussian of the summed error."""
    error = sum(abs(g - t) for g, t in zip(sorted(gaps), sorted(targets)))
    return math.exp(-(error / width) ** 2)


def _arrangement(hues, targets):
    """Best closeness of any len(targets) of ``hues`` to the cyclic gaps ``targets``."""
    k = len(targets)
    if len(hues) < k:
        return 0.0
    best = 0.0
    for subset in itertools.combinations(sorted(hues), k):
        gaps = [(subset[(i + 1) % k] - subset[i]) % 360.0 for i in range(k)]
        best = max(best, _closeness(gaps, targets))
    return best


def harmony(colors) -> list:
    """HARMONY scores in [0, 1] of a palette: what a decorator's color theory measures."""
    lch = [_LCH_of(c["color"]) for c in colors]
    chromatic = [h for L, C, h in lch if C >= NEUTRAL_CHROMA]
    weights = [C for L, C, h in lch if C >= NEUTRAL_CHROMA]
    pairs = list(itertools.combinations(chromatic, 2))
    gaps = [_hue_gap(a, b) for a, b in pairs]
    mono = 1.0 if len(chromatic) <= 1 else math.exp(-(max(gaps) / 30.0) ** 2)
    analogous = 1.0 if not gaps else sum(g <= 45.0 for g in gaps) / len(gaps)
    complementary = max((math.exp(-((g - 180.0) / 20.0) ** 2) for g in gaps), default=0.0)
    triadic = _arrangement(chromatic, (120.0, 120.0, 120.0))
    split = _arrangement(chromatic, (150.0, 60.0, 150.0))
    tetradic = max(_arrangement(chromatic, (90.0, 90.0, 90.0, 90.0)),
                   _arrangement(chromatic, (60.0, 120.0, 60.0, 120.0)))
    if weights:
        x = sum(w * math.cos(math.radians(h)) for w, h in zip(weights, chromatic)) / sum(weights)
        y = sum(w * math.sin(math.radians(h)) for w, h in zip(weights, chromatic)) / sum(weights)
        spread = 1.0 - math.hypot(x, y)
        bins = [0] * 12
        for h in chromatic:
            bins[int(h // 30) % 12] += 1
        entropy = -sum(n / len(chromatic) * math.log(n / len(chromatic)) for n in bins if n) / math.log(6)
    else:
        spread = entropy = 0.0
    lightness = [L for L, C, h in lch]
    chroma = [C for L, C, h in lch]
    mean_l = sum(lightness) / len(lightness)
    mean_c = sum(chroma) / len(chroma)
    warm = sum(1 for L, C, h in lch if C >= NEUTRAL_CHROMA and (h <= 100.0 or h >= 330.0)) / len(lch)
    distinct = min((math.dist(oklab(a["color"]), oklab(b["color"])) for a, b in itertools.combinations(colors, 2)),
                   default=0.0)
    return [len(colors) / 6.0, mono, analogous, complementary, triadic, split, tetradic, spread, min(1.0, entropy),
            1.0 - len(chromatic) / len(lch),
            min(1.0, 2 * math.sqrt(sum((v - mean_l) ** 2 for v in lightness) / len(lightness))),
            max(lightness) - min(lightness), min(1.0, mean_c / 0.3),
            min(1.0, 4 * math.sqrt(sum((v - mean_c) ** 2 for v in chroma) / len(chroma))), warm,
            min(1.0, distinct / 0.5)]


def features(design, secondary=None) -> list:
    """FEATURES floats, about 0..1: per element, then the palette's harmony, then the page's structure."""
    design = upgrade(design, secondary)
    colors = design["colors"]
    dark = _resolve(design, "dark")
    light = float(design["light"])
    row = []
    for name, (_role, box, _dark, _light) in ELEMENTS.items():
        token = design["elements"][name]
        row += _lab_unit(colors[token["slot"]]["color"])
        row += [float(token["tone"]), float(token["chroma"]) / 1.2]
        row += _lab_unit(dark[name])
        row += [float(token[a]) / _GEOMETRY_MAX[a] for a in GEOMETRY]
        row += [(math.sin(light) + 1) / 2 if box else 0.0, (math.cos(light) + 1) / 2 if box else 0.0,
                1.0 if box else 0.0]
    row += harmony(colors)
    L = {name: oklab(dark[name])[0] for name in ("page", "surface", "raised", "button")}
    row += [contrast(dark["text"], dark["page"]) / 21.0, contrast(dark["text"], dark["surface"]) / 21.0,
            contrast(dark["link"], dark["surface"]) / 21.0, contrast(dark["button"], dark["page"]) / 21.0,
            abs(L["surface"] - L["page"]) * 4, abs(L["raised"] - L["surface"]) * 4]
    assert len(row) == FEATURES, (len(row), FEATURES)
    return row


def names(design, secondary=None) -> list:
    return [c["name"] for c in upgrade(design, secondary)["colors"]]
