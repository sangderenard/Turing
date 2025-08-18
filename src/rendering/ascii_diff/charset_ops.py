#!/usr/bin/env python3
"""Charset and charmap utilities for FontMapper."""
from __future__ import annotations

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# --- END HEADER ---

# --- Optional: Torch (soft fail) ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from typing import Iterable, Tuple, List


def list_printable_characters(font_path: str, font_size: int = 12, epsilon: int = 1) -> str:
    """Return printable characters for a font filtered by width variance."""
    font = TTFont(font_path)
    cmap = font.getBestCmap()
    unicode_ranges = [
        range(0x0020, 0x007E),
        range(0x00A0, 0x00FF),
        range(0x0100, 0x017F),
        range(0x0370, 0x03FF),
        range(0x0400, 0x04FF),
        range(0x0590, 0x05FF),
        range(0x0600, 0x06FF),
        range(0x0900, 0x097F),
        range(0x4E00, 0x9FFF),
        range(0x0E00, 0x0E7F),
        range(0x10A0, 0x10FF),
        range(0x1D00, 0x1D7F),
        range(0x2000, 0x206F),
        range(0x20A0, 0x20CF),
        range(0x2100, 0x214F),
    ]
    safe_printable_chars = [chr(c) for r in unicode_ranges for c in r if c in cmap]
    font_pil = ImageFont.truetype(font_path, font_size)
    images = {}
    widths = []
    for char in safe_printable_chars:
        size, _ = font_pil.font.getsize(char)
        img = Image.new("L", size, color=255)
        ImageDraw.Draw(img).text((0, 0), char, font=font_pil, fill=0)
        images[char] = np.array(img)
    unique_chars: List[str] = []
    unique_widths: List[int] = []
    seen = set()
    for char, img in images.items():
        tup = tuple(map(tuple, img))
        if tup not in seen:
            seen.add(tup)
            unique_chars.append(char)
            (w, _), _ = font_pil.font.getsize(char)
            
            unique_widths.append(w)
    mode_width = max(set(unique_widths), key=unique_widths.count)
    filtered_chars = [c for c, w in zip(unique_chars, unique_widths) if abs(w - mode_width) <= epsilon]
    return "".join(filtered_chars)


def generate_checkerboard_pattern(width: int, height: int, block_size: int = 2) -> np.ndarray:
    """Create a checkerboard pattern for variant generation."""
    pattern = np.zeros((height, width), dtype=np.uint8)
    light_gray = 191
    dark_gray = 64
    for y in range(0, height, block_size * 2):
        for x in range(0, width, block_size * 2):
            pattern[y:y + block_size, x:x + block_size] = light_gray
            pattern[y + block_size:y + block_size * 2, x + block_size:x + block_size * 2] = light_gray
            pattern[y + block_size:y + block_size * 2, x:x + block_size] = dark_gray
            pattern[y:y + block_size, x + block_size:x + block_size * 2] = dark_gray
    return pattern


def generate_variants(
    charset: Iterable[str],
    fonts: Iterable[ImageFont.FreeTypeFont],
    max_width: int,
    max_height: int,
    level: int,
) -> List[np.ndarray]:
    """Generate bitmap variants for each character at a complexity level."""
    char_bitmasks: List[np.ndarray] = []
    bg_colors = [0]
    text_colors = [255]
    if level >= 2:
        bg_colors += [255]
        text_colors += [0]
    if level >= 3:
        bg_colors += [int(255 * 0.75), int(255 * 0.25)]
        text_colors += [int(255 * 0.25), int(255 * 0.75)]
    if level >= 4:
        bg_colors += [0, int(255 * 0.95)]
        text_colors += [int(255 * 0.05), 255]
    for char in charset:
        for font in fonts:
            (width, height), (off_x, off_y) = font.font.getsize(char)
            x_pos = (max_width - width) // 2
            y_pos = (max_height - (height + off_y)) // 2
            for bg, fg in zip(bg_colors, text_colors):
                img = Image.new("L", (max_width, max_height), color=bg)
                ImageDraw.Draw(img).text((x_pos, y_pos), char, font=font, fill=fg)
                char_bitmasks.append(np.array(img))
            if level >= 5:
                check = generate_checkerboard_pattern(max_width, max_height)
                img = Image.fromarray(check).convert("L")
                ImageDraw.Draw(img).text((x_pos, y_pos), char, font=font, fill=255)
                char_bitmasks.append(np.array(img))
    return char_bitmasks


def bytemaps_as_ascii(
    char_bitmasks: Iterable[np.ndarray | "torch.Tensor"],
    width: int,
    height: int,
    console_width: int = 240,
    ascii_gradient: str = " .:-=+*#%@",
) -> str:
    """Return ASCII preview for a collection of character bitmaps."""
    per_line = console_width // width
    lines_per_bm = height
    lines = ""
    bitmasks = list(char_bitmasks)
    for i in range(0, len(bitmasks), per_line):
        row = bitmasks[i:i + per_line]
        for line_no in range(lines_per_bm):
            for bitmask in row:
                if TORCH_AVAILABLE and "torch" in str(type(bitmask)):
                    bitmask = bitmask.squeeze().cpu().numpy()
                if bitmask.max() > 1:
                    bitmask = bitmask / 255.0
                for j in range(width):
                    val = bitmask[line_no, j]
                    idx = int(val * (len(ascii_gradient) - 1))
                    lines += ascii_gradient[idx]
                lines += " "
            lines += "\n"
        lines += "\n"
    return lines


def obtain_charset(
    font_files: List[str],
    font_size: int,
    complexity_level: int = 0,
    preset_charset: str | None = None,
) -> Tuple[List[ImageFont.FreeTypeFont], str, List[np.ndarray], int, int]:
    """Build charset and variants using provided fonts."""
    charset = preset_charset or list_printable_characters(font_files[0], font_size)
    max_w = 0
    max_h = 0
    fonts: List[ImageFont.FreeTypeFont] = []
    for path in font_files:
        font = ImageFont.truetype(path, font_size)
        for ch in charset:
            (w, h), (off_x, off_y) = font.font.getsize(ch)
            max_w = max(max_w, w)
            max_h = max(max_h, h + abs(off_y))
        fonts.append(font)
    bitmasks = generate_variants(charset, fonts, max_w, max_h, complexity_level)
    return fonts, charset, bitmasks, max_w, max_h


__all__ = [
    "list_printable_characters",
    "generate_checkerboard_pattern",
    "generate_variants",
    "bytemaps_as_ascii",
    "obtain_charset",
]
