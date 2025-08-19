import datetime as _dt
import time
import numpy as np
from PIL import Image

from .clock_renderer import ClockRenderer
from .theme_manager import ThemeManager
from .draw import get_changed_subunits, draw_diff, default_subunit_batch_to_chars
from .console import full_clear_and_reset_cursor, reset_cursor_to_top


def main() -> None:
    """Render a simple analog clock in the terminal using ASCII diff."""
    theme_manager = ThemeManager()
    # use default theme's active units
    units = theme_manager.current_theme.active_time_units
    # choose pixel diameter that maps nicely to character cells
    char_h = char_w = 16
    canvas_size = char_h * 12

    old_frame = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    full_clear_and_reset_cursor()

    while True:
        now = _dt.datetime.utcnow()
        img = ClockRenderer.render_analog(
            now,
            units=units,
            canvas_size_px=canvas_size,
            theme_manager=theme_manager,
        )
        frame = np.array(img.convert("RGB"))
        changed = get_changed_subunits(old_frame, frame, char_h, char_w)
        draw_diff(
            changed,
            char_cell_pixel_height=char_h,
            char_cell_pixel_width=char_w,
            subunit_to_char_kernel=default_subunit_batch_to_chars,
            active_ascii_ramp=theme_manager.get_current_ascii_ramp(),
        )
        old_frame = frame
        reset_cursor_to_top()
        time.sleep(1)


if __name__ == "__main__":
    main()
