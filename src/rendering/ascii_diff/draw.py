"""Terminal drawing helpers for ASCII frame diffs."""
from __future__ import annotations

import sys
import numpy as AbstractTensor
from colorama import Style, Fore, Back
from pathlib import Path
import logging
import os
from .ascii_kernel_classifier import AsciiKernelClassifier
from src.common.tensors.abstraction import AbstractTensor
test_tensor = AbstractTensor.get_tensor([0])
integer_type = test_tensor.long_dtype_


logger = logging.getLogger(__name__)
if os.getenv("TURING_DEBUG"):
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(_h)
    logger.setLevel(logging.DEBUG)
else:
    logger.addHandler(logging.NullHandler())

# Default ASCII ramp used if no specific ramp is provided to drawing functions.
DEFAULT_DRAW_ASCII_RAMP = " .:░▒▓█"


def flexible_subunit_kernel(
    subunit_data: AbstractTensor,
    ramp: str,
    mode: str = "ascii",
) -> str | AbstractTensor:
    """Return a representation for ``subunit_data`` based on ``mode``.

    Parameters
    ----------
    subunit_data:
        Pixel array representing one character cell. Shape ``(H, W, 3)``.
    ramp:
        ASCII ramp string ordered from darkest to brightest.
    mode:
        ``"ascii"``  -> return a single character.
        ``"raw"``    -> return the pixel array untouched.
        ``"hybrid"`` -> return a small image of the ASCII character.
    """
    if mode == "raw":
        return subunit_data
    char = default_subunit_batch_to_chars(
        AbstractTensor.expand_dims(subunit_data, dim=0), ramp
    )[0]
    if mode == "ascii":
        return char
    if mode == "hybrid":
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return subunit_data
        h, w = subunit_data.shape[:2]
        img = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        font_size = max(1, min(w, h))
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), char, font=font)
        x = (w - (bbox[2] - bbox[0])) // 2
        y = (h - (bbox[3] - bbox[1])) // 2
        draw.text((x, y), char, fill=(255, 255, 255), font=font)
        return AbstractTensor.array(img)
    raise ValueError(f"Unknown mode: {mode}")

# Module-level cache for classifier and ramp
_classifier_cache = {
    "ramp_char_size_use_nn": None,  # Cache key will be a tuple (ramp, char_width, char_height, use_nn)
    "classifier": None,
}

# Module-relative default font path
DEFAULT_FONT_PATH = Path(__file__).with_name("consola.ttf")

def default_subunit_batch_to_chars(
    subunit_batch: AbstractTensor,
    ramp: str = DEFAULT_DRAW_ASCII_RAMP,
    char_width: int = 16, # Add parameters for desired char_size
    char_height: int = 32, # These will be the actual cell_w, cell_h from clock_demo
    *,
    use_nn: bool = True,
    epsilon: float = 1e-4,
    max_epochs: int = 50,
) -> list[str]:
    """Return characters for ``subunit_batch`` using a cached classifier."""
    # Removed: (subunit_height, subunit_width) = get_char_cell_dims(),
    # Removed: print(subunit_height, subunit_width)
    # Removed: exit()
    # Now use the passed-in char_width and char_height

    cache_key = (ramp, char_width, char_height, use_nn)
    if _classifier_cache["ramp_char_size_use_nn"] != cache_key or _classifier_cache["classifier"] is None:
        classifier = AsciiKernelClassifier(
            ramp,
            font_path=str(DEFAULT_FONT_PATH),
            font_size=16,
            char_size=(char_width, char_height),
            use_nn=use_nn,
            epsilon=epsilon,
            max_epochs=max_epochs,
        )
        
        _classifier_cache["ramp_char_size_use_nn"] = cache_key
        _classifier_cache["classifier"] = classifier
    else:
        classifier = _classifier_cache["classifier"]

    result = classifier.classify_batch(subunit_batch)
    return result["chars"]


def draw_text_overlay(
    row: int,
    col: int,
    text: str,
    color_code: str = Style.RESET_ALL # Default to terminal default color
) -> None:
    """Draws text directly to the terminal at a specific 1-indexed row and column."""
    # Use \x1b[<row>;<col>H for cursor positioning
    sys.stdout.write(f"\x1b[{row};{col}H{color_code}{text}{Style.RESET_ALL}")
    sys.stdout.flush()

def draw_diff(
    changed_subunits: list[tuple[int, int, AbstractTensor]],  # List of (y_pixel, x_pixel, subunit_pixel_data)
    char_cell_pixel_height: int = 1,  # The height of a character cell in pixels
    char_cell_pixel_width: int = 1,  # The width of a character cell in pixels
    subunit_to_char_kernel: callable[[AbstractTensor, str, int, int], list[str]] = default_subunit_batch_to_chars, # Update signature
    active_ascii_ramp: str = DEFAULT_DRAW_ASCII_RAMP,  # The ASCII ramp to use for character conversion
    base_row: int = 1, 
    base_col: int = 1,
    enable_fg_color: bool = True,
    enable_bg_color: bool = True,
) -> None:
    """Render changed subunits using a kernel to map subunit data to characters.
    
    `y_pixel`, `x_pixel` in `changed_subunits` are 0-indexed top-left pixel coordinates of the subunit.
    `char_cell_pixel_height` and `char_cell_pixel_width` define the dimensions of a single
    character cell in terms of pixels. The `subunit_data` for each entry in `changed_subunits`
    is expected to match these dimensions (or be a part of a larger image from which these
    coordinates are derived). The `subunit_to_char_kernel` is expected to accept the entire
    batch of subunits and return a list of ASCII characters for display.
    `base_row`, `base_col` are 1-indexed for ANSI terminal compatibility.
    """
    if char_cell_pixel_height <= 0: char_cell_pixel_height = 1
    if char_cell_pixel_width <= 0: char_cell_pixel_width = 1
    if not changed_subunits:
        logger.debug("No changed subunits to draw")
        return
    subunit_batch = AbstractTensor.stack([data for _, _, data in changed_subunits], dim=0)
    # Pass the actual char_cell_pixel_width and char_cell_pixel_height to the kernel
    chars = subunit_to_char_kernel(subunit_batch, active_ascii_ramp, char_cell_pixel_width, char_cell_pixel_height)
    logger.debug("Drawing %d changed subunits", len(changed_subunits))
    for (y_pixel, x_pixel, subunit_data), char_to_draw in zip(changed_subunits, chars):
        char_y = y_pixel // char_cell_pixel_height
        char_x = x_pixel // char_cell_pixel_width

        ansi_color_bg = ""
        ansi_color_fg = ""
        if subunit_data.ndim == 3 and subunit_data.shape[2] == 3:
            avg_color = AbstractTensor.mean(subunit_data.to_dtype(test_tensor.float_dtype), dim=(0, 1)).to_dtype(integer_type)
            r, g, b = avg_color[0], avg_color[1], avg_color[2]
            if enable_bg_color:
                ansi_color_bg = f"\x1b[48;2;{r.item()};{g.item()};{b.item()}m"
            if enable_fg_color:
                ansi_color_fg = "\x1b[38;2;255;255;255m"
        else:
            if enable_fg_color:
                ansi_color_fg = "\x1b[38;2;255;255;255m"

        terminal_row = base_row + char_y
        terminal_col = base_col + char_x

        sys.stdout.write(
            f"\x1b[{terminal_row};{terminal_col}H{ansi_color_bg}{ansi_color_fg}{char_to_draw}\x1b[0m"
        )
    sys.stdout.flush()


from typing import List, Tuple

def get_changed_subunits(
    old_frame: AbstractTensor,
    new_frame: AbstractTensor,
    subunit_height: int,
    subunit_width: int,
    loss_threshold: float = 0.0,
) -> List[Tuple[int, int, AbstractTensor]]:
    """
    Compares two frames and returns subunits from the new_frame that have changed.

    The frames are expected to be NumPy arrays, typically representing image data
    with shape (height, width) or (height, width, channels).

    Args:
        old_frame: The previous frame.
        new_frame: The current frame.
        subunit_height: The height of the rectangular subunits to check.
        subunit_width: The width of the rectangular subunits to check.
        loss_threshold:
            Mean absolute difference threshold for considering a subunit
            changed. A value of ``0.0`` means any difference will be
            returned.

    Returns:
        A list of tuples. Each tuple contains:
        - y_coord (int): The y-coordinate of the top-left corner of the changed subunit
                         in the original frame.
        - x_coord (int): The x-coordinate of the top-left corner of the changed subunit
                         in the original frame.


    Raises:
        ValueError: If old_frame and new_frame do not have the same shape,
                    or if subunit dimensions are not positive.
    """
    if old_frame.shape != new_frame.shape:
        raise ValueError("Old and new frames must have the same shape.")
    if not (subunit_height > 0 and subunit_width > 0):
        raise ValueError("Subunit height and width must be positive integers.")

    frame_height, frame_width = new_frame.shape[0], new_frame.shape[1]
    changed_subunits_list: List[Tuple[int, int, AbstractTensor]] = []

    # Vectorized difference computation for regions that align to the subunit
    # grid. Any leftover edge regions fall back to the original loop logic.
    h_full = frame_height - (frame_height % subunit_height)
    w_full = frame_width - (frame_width % subunit_width)

    if h_full > 0 and w_full > 0:
        diff = AbstractTensor.abs(
            old_frame[:h_full, :w_full].to_dtype(integer_type)
            - new_frame[:h_full, :w_full].to_dtype(integer_type)
        )
        reshaped = diff.reshape(
            h_full // subunit_height,
            subunit_height,
            w_full // subunit_width,
            subunit_width,
            -1,
        )
        losses = reshaped.mean(dim=(1, 3, 4))
        coords = AbstractTensor.argwhere(losses > loss_threshold)
        for by, bx in coords:
            y = int(by * subunit_height)
            x = int(bx * subunit_width)
            changed_subunits_list.append(
                (y, x, new_frame[y : y + subunit_height, x : x + subunit_width].copy())
            )

    # Handle right and bottom edges that may be smaller than the subunit size
    for y in range(0, frame_height, subunit_height):
        for x in range(w_full, frame_width, subunit_width):
            y_end = min(y + subunit_height, frame_height)
            x_end = min(x + subunit_width, frame_width)
            changed_subunits_list.append((y, x, new_frame[y:y_end, x:x_end].copy()))

    for y in range(h_full, frame_height, subunit_height):
        for x in range(0, w_full, subunit_width):
            y_end = min(y + subunit_height, frame_height)
            x_end = min(x + subunit_width, frame_width)
            changed_subunits_list.append((y, x, new_frame[y:y_end, x:x_end].copy()))

    for y in range(h_full, frame_height, subunit_height):
        for x in range(w_full, frame_width, subunit_width):
            y_end = min(y + subunit_height, frame_height)
            x_end = min(x + subunit_width, frame_width)
            changed_subunits_list.append((y, x, new_frame[y:y_end, x:x_end].copy()))

    return changed_subunits_list
