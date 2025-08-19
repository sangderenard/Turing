#!/usr/bin/env python3
"""ASCII art utilities for rendering clock faces."""
from __future__ import annotations

import datetime as _dt
import math
from typing import List, Tuple, Optional
# --- END HEADER ---

from colorama import Fore, Style
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
    from .theme_manager import ThemeManager
    from .render_backend import RenderingBackend
except ImportError:
    PIL_AVAILABLE = False

# ASCII ramp for gradient effect (more dense = brighter/more opaque)
ASCII_RAMP_BLOCK = " .:░▒▓█" # Using block characters for a more solid look
ASCII_RAMP_DETAILED = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

from .time_units import TimeUnit # Import TimeUnit
# Palette of Colorama colors and their approximate RGB values
# (R, G, B) values are 0-255
COLORAMA_PALETTE: List[Tuple[str, Tuple[int, int, int]]] = [
    (Fore.BLACK, (0, 0, 0)),
    (Fore.RED, (128, 0, 0)),
    (Fore.GREEN, (0, 128, 0)),
    (Fore.YELLOW, (128, 128, 0)),
    (Fore.BLUE, (0, 0, 128)),
    (Fore.MAGENTA, (128, 0, 128)),
    (Fore.CYAN, (0, 128, 128)),
    (Fore.WHITE, (192, 192, 192)), # Often light gray
    (Fore.LIGHTBLACK_EX, (128, 128, 128)), # Often dark gray
    (Fore.LIGHTRED_EX, (255, 0, 0)),
    (Fore.LIGHTGREEN_EX, (0, 255, 0)),
    (Fore.LIGHTYELLOW_EX, (255, 255, 0)),
    (Fore.LIGHTBLUE_EX, (0, 0, 255)),
    (Fore.LIGHTMAGENTA_EX, (255, 0, 255)),
    (Fore.LIGHTCYAN_EX, (0, 255, 255)),
    (Fore.LIGHTWHITE_EX, (255, 255, 255)),
]
# For convenience, a "bright" version of the palette using Style.BRIGHT
COLORAMA_PALETTE_BRIGHT: List[Tuple[str, Tuple[int, int, int]]] = [
    (Style.BRIGHT + Fore.BLACK, (128, 128, 128)), # Bright black is grey
    (Style.BRIGHT + Fore.RED, (255, 0, 0)),
    (Style.BRIGHT + Fore.GREEN, (0, 255, 0)),
    (Style.BRIGHT + Fore.YELLOW, (255, 255, 0)),
    (Style.BRIGHT + Fore.BLUE, (0, 0, 255)),
    (Style.BRIGHT + Fore.MAGENTA, (255, 0, 255)),
    (Style.BRIGHT + Fore.CYAN, (0, 255, 255)),
    (Style.BRIGHT + Fore.WHITE, (255, 255, 255)),
]
# We'll primarily use the bright palette for more vibrant foregrounds.
ACTIVE_COLOR_PALETTE = COLORAMA_PALETTE_BRIGHT

# Keep old digits for a potential fallback if PIL is not available, though not used in current impl.
_DIGITS_OLD = {
    '0': [
        " ### ",
        "#   #",
        "#   #",
        "#   #",
        " ### ",
    ],
    '1': [
        "  #  ",
        " ##  ",
        "  #  ",
        "  #  ",
        " ### ",
    ],
    '2': [
        " ### ",
        "    #",
        " ### ",
        "#    ",
        "#####",
    ],
    '3': [
        "#####",
        "    #",
        " ### ",
        "    #",
        "#####",
    ],
    '4': [
        "#   #",
        "#   #",
        "#####",
        "    #",
        "    #",
    ],
    '5': [
        "#####",
        "#    ",
        "#### ",
        "    #",
        "#### ",
    ],
    '6': [
        " ### ",
        "#    ",
        "#### ",
        "#   #",
        " ### ",
    ],
    '7': [
        "#####",
        "    #",
        "   # ",
        "  #  ",
        "  #  ",
    ],
    '8': [
        " ### ",
        "#   #",
        " ### ",
        "#   #",
        " ### ",
    ],
    '9': [
        " ### ",
        "#   #",
        " ####",
        "    #",
        " ### ",
    ],
    ':': [
        "     ",
        "  #  ",
        "     ",
        "  #  ",
        "     ",
    ],
    '.': [
        "     ",
        "     ",
        "     ",
        "     ",
        "  #  ",
    ],
    ' ': [
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
    ],
}

def _compose_digits_fallback(text: str) -> list[str]:
    """Compose ASCII rows for ``text`` using ``_DIGITS_OLD``."""
    grids = [_DIGITS_OLD.get(ch, _DIGITS_OLD[' ']) for ch in text]
    rows = []
    for i in range(len(_DIGITS_OLD['0'])):
        row = ' '.join(grid[i] for grid in grids)
        rows.append(row)
    return rows


def _ascii_rows_to_pixel(rows: list[str]) -> np.ndarray:
    """Convert ASCII rows to a white pixel array."""
    h = len(rows)
    w = len(rows[0]) if rows else 0
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ' ':
                arr[y, x] = [255, 255, 255]
    return arr


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont | None:
    if not PIL_AVAILABLE:
        return None
    common_fonts = ["DejaVuSansMono.ttf", "cour.ttf", "Consolas.ttf", "Menlo.ttf", "LiberationMono-Regular.ttf"]
    for font_name in common_fonts:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    try: # Fallback to a generic sans-serif if monospace not found
        return ImageFont.truetype("arial.ttf", size)
    except IOError: # PIL's default bitmap font if no ttf found
        try:
            return ImageFont.load_default() # This font is very small
        except IOError:
            return None


def _find_closest_color(rgb_pixel: Tuple[int, int, int], palette: List[Tuple[str, Tuple[int, int, int]]]) -> str:
    """Finds the closest color in the palette to the given RGB pixel."""
    if not palette:
        return Fore.WHITE # Fallback
    
    min_dist = float('inf')
    closest_color_code = palette[0][0]
    r1, g1, b1 = rgb_pixel

    for color_code, (r2, g2, b2) in palette:
        dist = (r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2
        if dist < min_dist:
            min_dist = dist
            closest_color_code = color_code
    return closest_color_code

def _image_to_ascii_colored(
    image: Image.Image, # Expects RGBA image
    ramp: str = ASCII_RAMP_BLOCK,
    ascii_bg_fill: str = Fore.BLACK + Style.DIM + " ", # Colorama string for background char
    target_ascii_width: int = 20,
    target_ascii_height: int = 14,
    bg_alpha_threshold: int = 10, # Alpha below this is considered background
) -> str:
    if not PIL_AVAILABLE:
        return "Pillow library not found."

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Resize image so each pixel corresponds to an ASCII character cell
    img_for_ascii = image.resize((target_ascii_width, target_ascii_height), Image.Resampling.LANCZOS)

    ascii_art_rows = []
    ramp_len = len(ramp)

    for y_idx in range(img_for_ascii.height):
        line_chars = []
        for x_idx in range(img_for_ascii.width):
            r, g, b, a = img_for_ascii.getpixel((x_idx, y_idx))

            if a < bg_alpha_threshold:
                # Background pixel (transparent or nearly transparent)
                colored_char = ascii_bg_fill[0:-1] + ramp[0] + Style.RESET_ALL if ascii_bg_fill else ramp[0]
            else:
                # Foreground pixel
                char_color_code = _find_closest_color((r, g, b), ACTIVE_COLOR_PALETTE)
                luminance = int(0.299 * r + 0.587 * g + 0.114 * b) # Standard luminance calc
                char_index = min(ramp_len - 1, int((luminance / 255) * ramp_len))
                char = ramp[char_index]
                colored_char = char_color_code + char + Style.RESET_ALL
            line_chars.append(colored_char)
        ascii_art_rows.append("".join(line_chars))
    return "\n".join(ascii_art_rows)


def compose_ascii_digits(
    text: str,
    font_size: int = 32,
    target_ascii_width: int = 60,
    target_ascii_height: int = 5,
    text_color_on_image: Tuple[int,int,int,int] = (255,255,255,255), # White opaque for main text
    shadow_color_on_image: Tuple[int,int,int,int] = (0,0,0,100), # Semi-transparent black for shadow
    outline_color_on_image: Tuple[int,int,int,int] = (0,0,0,200), # Opaque black for outline
    outline_thickness: int = 1,
    shadow_offset: Tuple[int, int] = (2, 2), # (dx, dy) for shadow from text
    backdrop_image_path: Optional[str] = None,
    final_ascii_bg_fill: str = Fore.BLACK + Style.DIM + " ", # Used for transparent areas
    *,
    as_pixel_array: bool = False,
    target_pixel_width: Optional[int] = None,
    target_pixel_height: Optional[int] = None,
    ascii_ramp: str = ASCII_RAMP_BLOCK,
    theme_manager: Optional[ThemeManager] = None,
    render_backend: Optional[RenderingBackend] = None,
    # Parameters for when compose_ascii_digits is used for arbitrary text, not just time
    # These would typically be simpler than full clock effects.
) -> str | np.ndarray:
    if not PIL_AVAILABLE:
        rows = _compose_digits_fallback(text)
        if as_pixel_array:
            return _ascii_rows_to_pixel(rows)
        return "\n".join(rows)

    font = _get_font(font_size)
    if not font: # Handle case where load_default also fails
        return f"Font not found (size {font_size}). Text: {text}"

    # Determine text bounding box to size the image correctly
    try: # For TrueType/FreeType fonts
        text_bbox = font.getbbox(text)
        text_x_offset, text_y_offset = text_bbox[0], text_bbox[1]
        text_render_width, text_render_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    except AttributeError: # Fallback for bitmap fonts (like load_default())
        text_render_width, text_render_height = font.getsize(text) # Deprecated but works for bitmap
        text_x_offset, text_y_offset = 0, 0 # Bitmap fonts usually render from (0,0) relative to anchor

    # Image dimensions need to accommodate text and shadow
    img_width = text_render_width + abs(shadow_offset[0]) + outline_thickness * 2 + 4 # Extra padding
    img_height = text_render_height + abs(shadow_offset[1]) + outline_thickness * 2 + 4 # Extra padding

    # Create base image: either backdrop or new transparent image
    if backdrop_image_path:
        try:
            image = Image.open(backdrop_image_path).convert("RGBA")
            image = image.resize((img_width, img_height), Image.Resampling.LANCZOS)
        except FileNotFoundError:
            return f"Backdrop image not found: {backdrop_image_path}"
        except Exception as e:
            return f"Error loading backdrop: {e}"
    else:
        image = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Position for drawing text (top-left of the text rendering area)
    # Adjust for shadow offset: if shadow is positive, text needs to be at (0,0) relative to its box
    # and shadow at (offset_x, offset_y). If shadow is negative, text needs to be at (abs(offset_x), abs(offset_y))
    # Also adjust for outline thickness
    padding = 2 + outline_thickness
    base_x = padding + (abs(shadow_offset[0]) if shadow_offset[0] < 0 else 0) - text_x_offset
    base_y = padding + (abs(shadow_offset[1]) if shadow_offset[1] < 0 else 0) - text_y_offset

    # 1. Draw Shadow
    shadow_pos_x = base_x + shadow_offset[0]
    shadow_pos_y = base_y + shadow_offset[1]
    # Draw shadow with its own (potentially blurred) outline if desired, or just plain
    if shadow_color_on_image[3] > 0: # Only draw if shadow is visible
        draw.text((shadow_pos_x, shadow_pos_y), text, font=font, fill=shadow_color_on_image)

    # 2. Draw Main Text with Outline (on top of shadow)
    # Pillow's stroke is centered on the text path, so text is drawn over half the stroke.
    if isinstance(font, ImageFont.FreeTypeFont) and outline_thickness > 0 and outline_color_on_image[3] > 0:
        draw.text((base_x, base_y), text, font=font, fill=text_color_on_image,
                  stroke_width=outline_thickness, stroke_fill=outline_color_on_image)
    else: # Fallback for bitmap fonts or no outline
        draw.text((base_x, base_y), text, font=font, fill=text_color_on_image)

    if render_backend:
        image = render_backend.process(image)
        if theme_manager:
            ascii_ramp = theme_manager.get_current_ascii_ramp()
    elif theme_manager:
        image = theme_manager.apply_effects(image)
        image = theme_manager.apply_theme(image)
        ascii_ramp = theme_manager.get_current_ascii_ramp()

    if as_pixel_array:
        px_w = target_pixel_width or target_ascii_width
        px_h = target_pixel_height or target_ascii_height
        return np.array(
            image.resize((px_w, px_h), Image.Resampling.LANCZOS).convert("RGB")
        )

    return _image_to_ascii_colored(
        image,
        ramp=ascii_ramp,
        ascii_bg_fill=final_ascii_bg_fill,
        target_ascii_width=target_ascii_width,
        target_ascii_height=target_ascii_height,
    )


def render_ascii_to_array(text: str, **kwargs) -> np.ndarray:
    """Return ASCII art for ``text`` as a ``numpy.ndarray``."""
    ascii_str = compose_ascii_digits(text, **kwargs)
    rows = ascii_str.splitlines()
    return np.array([list(row) for row in rows], dtype="<U1")
