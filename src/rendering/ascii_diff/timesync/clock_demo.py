#!/usr/bin/env python3
"""Demonstrate time synchronization with analog and digital clocks.

The demo runs until ``q`` or ``quit`` is received on standard input.
"""
from __future__ import annotations


#from AGENTS.tools.headers.header_utils import ENV_SETUP_BOX
import datetime as _dt
import argparse
import time
import json
import os
import threading
import sys
import numpy as np
from colorama import Style, Fore, Back  # For colored terminal output
from timesync import (
        get_offset,
        sync_offset,
        init_colorama_for_windows,
        reset_cursor_to_top,
        full_clear_and_reset_cursor,
    )
from timesync.timesync.theme_manager import (
        ThemeManager,
        ClockTheme,
    )
from timesync.timesync.render_backend import RenderingBackend
from timesync.frame_buffer import PixelFrameBuffer
from timesync.render_thread import render_loop
from timesync.draw import draw_diff
from timesync.timesync.ascii_digits import (
        ASCII_RAMP_BLOCK,
    )
from timesync.timesync.clock_renderer import ClockRenderer
from timesync.draw import draw_text_overlay  # Import the new text drawing function
from PIL import Image
import queue
from timesync.menu_resolver import MenuResolver

    # Platform-specific input handling (adapted from AGENTS/tools/dev_group_menu.py)
if os.name == "nt":  # Windows
    pass  # Windows specific input will be handled directly in input_thread_fn
else:  # Unix-like
    import select
    import sys
    import termios
    import tty

    def getch_timeout(timeout_seconds: float) -> str | None:
        """Get a single character with timeout on Unix-like systems."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ready_to_read, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
            if ready_to_read:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# --- END HEADER ---




def parse_norm_rect(rect_str: str) -> tuple[float, float, float, float]:
    """
    Parses 'x,y,width,height' as normalized floats in [0..1].
    Example: '0.1,0.2,0.5,0.4' => (0.1, 0.2, 0.5, 0.4).
    Clamps values between 0.0 and 1.0.
    """
    try:
        parts = [float(p.strip()) for p in rect_str.split(",")]
        if len(parts) == 4:
            # clamp each value to [0.0..1.0]
            parts = [max(0.0, min(1.0, v)) for v in parts]
            return tuple(parts)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"Invalid normalized rect: '{rect_str}'.")


def load_config_from_json(
    config_path: str, clock_identifier: Optional[str] = None
) -> dict:
    """Loads configuration from a JSON file."""
    if clock_identifier:
        config_path = f"{clock_identifier}_config.json"

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {config_path}. Using defaults.")
        except Exception as e:
            print(f"Warning: Error loading {config_path}: {e}. Using defaults.")
    return {}


def save_config_to_json(config_data: dict, clock_identifier: str) -> None:
    """Saves configuration to a JSON file."""
    config_path = f"{clock_identifier}_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration to {config_path}: {e}")


def interactive_configure_mode(
    clock_identifier: str,  # e.g., "slow_analog", "fast_digital"
    initial_config: dict,
    theme_manager: ThemeManager,  # Pass theme_manager for backdrop access
    key_mappings: dict,
):
    """Interactive mode for configuring clock appearance."""
    config = initial_config.copy()

    clock_type = "analog" if "analog" in clock_identifier else "digital"
    is_fast_clock = "fast" in clock_identifier

    print(f"Entering {clock_identifier} clock configuration mode...")
    current_backdrop = theme_manager.current_theme.current_backdrop_path
    print(f"Using backdrop: {current_backdrop if current_backdrop else 'None'}")
    config_mappings = {
        action: mapping
        for action, mapping in key_mappings.items()
        if action.startswith("config_")
    }
    legend_parts = [
        f"{'/'.join(m['keys'])}: {m['description']}" for m in config_mappings.values()
    ]
    print("Press the following keys:")
    for part in legend_parts:
        print(part)
    time.sleep(1)

    adjustment_step = 2

    while True:
        full_clear_and_reset_cursor()
        current_time = _dt.datetime.utcnow().replace(
            tzinfo=_dt.timezone.utc
        )  # Or use timesync.now()
        stopwatch_td = _dt.timedelta(
            seconds=time.perf_counter()
        )  # Dummy stopwatch for config

        time_obj_for_config = stopwatch_td if is_fast_clock else current_time

        print(f"--- {clock_identifier.upper()} CLOCK CONFIGURATION ---")
        if clock_type == "analog":
            # Ensure rect is a tuple if it exists
            rect = config.get("clock_drawing_rect_on_canvas")
            if isinstance(rect, list) and len(rect) == 4:
                config["clock_drawing_rect_on_canvas"] = tuple(rect)

            # Use theme_manager's current backdrop for preview
            print_analog_clock(
                time_obj_for_config,
                backdrop_image_path=theme_manager.current_theme.current_backdrop_path,
                theme_manager=theme_manager,
                **config,
            )
            print("\n--- Current Analog Config ---")
            rect = config.get("clock_drawing_rect_on_canvas", "Not set")
            print(f"Drawing Rect (x,y,w,h): {rect}")
            print(
                f"ASCII Diameter: {config.get('target_ascii_diameter', 'Default')}"
            )
            print(f"Canvas Size PX: {config.get('canvas_size_px', 'Default')}")

        elif clock_type == "digital":
            print_digital_clock(
                time_obj_for_config,
                backdrop_image_path=theme_manager.current_theme.current_backdrop_path,
                theme_manager=theme_manager,
                **config,
            )
            print("\n--- Current Digital Config ---")
            print(f"ASCII Width: {config.get('target_ascii_width', 'Default')} (←/→)")
            print(f"ASCII Height: {config.get('target_ascii_height', 'Default')} (↑/↓)")
            print(f"Font Size: {config.get('font_size', 'Default')} (+/-)")

        print()
        for part in legend_parts:
            print(part)

        key = getch_timeout(0.15)  # Shorter timeout for better responsiveness
        if key is None:
            continue

        action = None
        if key == "\x1b":
            next_key1 = getch_timeout(0.1)
            if next_key1 == "[":
                next_key2 = getch_timeout(0.1)
                arrow_map = {"D": "LEFT", "C": "RIGHT", "A": "UP", "B": "DOWN"}
                token = arrow_map.get(next_key2)
                if token:
                    for act, mapping in config_mappings.items():
                        if token in mapping["keys"]:
                            action = act
                            break
        else:
            for act, mapping in config_mappings.items():
                if key in mapping["keys"]:
                    action = act
                    break

        if action is None:
            continue

        if action == "config_quit":
            break
        elif action == "config_save":
            save_config_to_json(config, clock_identifier)
            print("Configuration saved. Exiting config mode.")
            time.sleep(1)
            break

        if clock_type == "analog":
            rect = list(
                config.get("clock_drawing_rect_on_canvas", [20, 20, 80, 80])
            )
            if action == "config_left":
                rect[0] -= adjustment_step
            elif action == "config_right":
                rect[0] += adjustment_step
            elif action == "config_up":
                rect[1] -= adjustment_step
            elif action == "config_down":
                rect[1] += adjustment_step
            elif action == "config_width_increase":
                rect[2] += adjustment_step
            elif action == "config_width_decrease":
                rect[2] -= adjustment_step
            elif action == "config_height_decrease":
                rect[3] -= adjustment_step
            elif action == "config_height_increase":
                rect[3] += adjustment_step

            config["clock_drawing_rect_on_canvas"] = tuple(max(0, val) for val in rect)

            if action == "config_ascii_diameter_increase":
                config["target_ascii_diameter"] = config.get("target_ascii_diameter", 22) + 1
            elif action == "config_ascii_diameter_decrease":
                config["target_ascii_diameter"] = max(
                    5, config.get("target_ascii_diameter", 22) - 1
                )
            elif action == "config_canvas_increase":
                config["canvas_size_px"] = config.get("canvas_size_px", 120) + 10
            elif action == "config_canvas_decrease":
                config["canvas_size_px"] = max(
                    20, config.get("canvas_size_px", 120) - 10
                )

        elif clock_type == "digital":
            if action == "config_left":
                config["target_ascii_width"] = max(
                    10, config.get("target_ascii_width", 60) - adjustment_step
                )
            elif action == "config_right":
                config["target_ascii_width"] = config.get("target_ascii_width", 60) + adjustment_step
            elif action == "config_up":
                config["target_ascii_height"] = max(
                    3, config.get("target_ascii_height", 7) - 1
                )
            elif action == "config_down":
                config["target_ascii_height"] = config.get("target_ascii_height", 7) + 1

            if action == "config_font_increase":
                config["font_size"] = config.get("font_size", 40) + adjustment_step
            elif action == "config_font_decrease":
                config["font_size"] = max(
                    8, config.get("font_size", 40) - adjustment_step
                )

    full_clear_and_reset_cursor()
    print(f"Exited {clock_identifier} clock configuration mode.")


def input_thread_fn(input_queue, stop_event):
    if os.name == "nt":  # Windows specific logic
        import msvcrt

        while not stop_event.is_set():
            if msvcrt.kbhit():  # Check if a key is pressed
                try:
                    key_bytes = msvcrt.getch()
                    # Handle special keys like arrows if necessary, they might be multi-byte
                    # For now, decode simply.
                    key = key_bytes.decode(errors="ignore")
                    if key:  # Ensure key is not empty or None before putting
                        input_queue.put(key)
                except Exception:
                    pass  # Ignore decode errors or other issues with getch
            time.sleep(0.01)  # Small sleep to prevent busy-waiting at 100% CPU
    else:  # Unix-like specific logic (uses the getch_timeout defined above for Unix)
        while not stop_event.is_set():
            key = getch_timeout(0.01)  # Poll frequently using the Unix getch_timeout
            if key:
                input_queue.put(key)


# Path to key mappings JSON
KEY_MAPPINGS_PATH = os.path.join(
    os.path.dirname(__file__), "timesync", "key_mappings.json"
)


# Or for more robustness, add debugging:
def load_key_mappings(path: str = KEY_MAPPINGS_PATH) -> dict:
    try:
        with open(path, "r") as f:
            mappings = json.load(f)
            print(f"Successfully loaded key mappings from {path}")
            return mappings
    except FileNotFoundError:
        print(f"Key mappings file not found at {path}")
        # Try alternate locations
        alt_paths = [
            os.path.join(os.path.dirname(__file__), "key_mappings.json"),
            os.path.join(os.path.dirname(__file__), "timesync", "key_mappings.json"),
            os.path.join(
                os.path.dirname(__file__), "..", "timesync", "key_mappings.json"
            ),
        ]
        for alt_path in alt_paths:
            try:
                with open(alt_path, "r") as f:
                    mappings = json.load(f)
                    print(f"Found key mappings at alternate path: {alt_path}")
                    return mappings
            except FileNotFoundError:
                continue
        print("Could not find key_mappings.json in any expected location")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing key mappings JSON from {path}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error loading key mappings from {path}: {e}")
        return {}





class InputDispatcher:
    """
    Dispatches input events to action handlers using MenuResolver.
    """

    def __init__(self, menu_resolver):
        self.menu_resolver = menu_resolver

    def dispatch(self, input_buffer):
        change_occured = False
        for char_key_raw in input_buffer:
            action = self.menu_resolver.handle_key(char_key_raw)
            if action:
                # Handle mode transitions for config
                if action == "enter_config_mode":
                    self.menu_resolver.push_mode("config")
                    self.menu_resolver.set_focused_clock_type("analog")  # or "digital"
                elif action == "config_quit":
                    self.menu_resolver.pop_mode()
                    self.menu_resolver.set_focused_clock_type(None)
                change_occured = True
        return change_occured

def main() -> None:
    """Run the clock demo until interrupted."""
    global PIXEL_BUFFER_SCALE, TEXT_FIELD_SCALE  # Allow modification by keys

    parser = argparse.ArgumentParser(
        description="Live clock demo with various displays."
    )
    parser.add_argument(
        "--configure",
        choices=["slow_analog", "fast_analog", "slow_digital", "fast_digital"],
        help="Enter interactive configuration mode for a specific clock instance.",
    )
    parser.add_argument(
        "--backdrops",
        nargs="+",
        type=str,
        help="Space-separated list of backdrop image paths to cycle through. The first is default.",
    )
    parser.add_argument(
        "--initial-pixel-buffer-scale",
        type=float,
        default=1.0,
        help="Initial scale factor for the internal render buffer resolution.",
    )
    parser.add_argument(
        "--initial-text-field-scale",
        type=float,
        default=1.0,
        help="Initial scale factor for text field output size (characters).",
    )
    parser.add_argument(
        "--analog-backdrop",
        type=str,
        help="Path to an image file for the analog clock backdrop.",
    )
    parser.add_argument(
        "--digital-backdrop",
        type=str,
        help="Path to an image file for digital clock backdrops.",
    )
    parser.add_argument(
        "--analog-clock-rect",
        type=parse_norm_rect,
        help="Normalized 'x,y,w,h' in [0..1] for analog clock placement on its canvas.",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=0.1,
        help="Refresh rate in seconds (e.g., 0.1 for 10 FPS).",
    )
    parser.add_argument(
        "--no-analog",
        action="store_false",
        dest="show_analog",
        help="Do not display the analog clock.",
    )
    parser.add_argument(
        "--no-digital-system",
        action="store_false",
        dest="show_digital_system",
        help="Do not display the digital system clock.",
    )
    parser.add_argument(
        "--no-digital-internet",
        action="store_false",
        dest="show_digital_internet",
        help="Do not display the digital internet clock.",
    )
    parser.add_argument(
        "--no-stopwatch",
        action="store_false",
        dest="show_stopwatch",
        help="Do not display the stopwatch.",
    )
    parser.add_argument(
        "--no-offset",
        action="store_false",
        dest="show_offset",
        help="Do not display the time offset.",
    )
    parser.add_argument(
        "--theme",
        choices=["cyberpunk", "matrix", "retro"],
        default="cyberpunk",
        help="Color theme preset",
    )
    parser.add_argument(
        "--effects",
        choices=[
            "neon",
            "crisp",
            "dramatic",
            "ethereal",
            "crystalline",
            "ghostly",
            "vivid",
            "soft_glow",
            "shadowed",
            "vintage_bright",
            "dreamlike",
            "sketch",
        ],
        default="neon",
        help="Visual effects preset",
    )
    parser.add_argument(
        "--post-processing",
        choices=[
            "high_contrast",
            "soft",
            "monochrome",
            "vintage",
            "neon_glow",
            "matrix_code",
            "cyberdream",
            "night_vision",
            "sepia_tone",
            "cold_wave",
            "hot_warm",
            "comic_book",
            "vibrant",
            "low_light",
        ],
        default="high_contrast",
        help="Post-processing preset",
    )
    parser.add_argument(
        "--ascii-style",
        choices=["block", "detailed", "minimal", "dots", "shapes"],
        default="detailed",
        help="Initial ASCII character style",  # Changed default to detailed
    )
    parser.add_argument(
        "--slow-analog-units",
        type=str,
        default="default_analog",
        help="Time unit set for the slow analog clock.",
    )
    parser.add_argument(
        "--fast-analog-units",
        type=str,
        default="analog_fast_sms",
        help="Time unit set for the fast analog clock.",
    )
    parser.add_argument(
        "--slow-digital-units",
        type=str,
        default="default_digital_hms",
        help="Time unit set for the slow digital clock.",
    )
    parser.add_argument(
        "--fast-digital-units",
        type=str,
        default="digital_fast_sms",
        help="Time unit set for the fast digital clock.",
    )
    parser.add_argument(
        "--slow-digital-format",
        type=str,
        default="default_hms",
        help="Format key for slow digital clock.",
    )
    parser.add_argument(
        "--fast-digital-format",
        type=str,
        default="fast_s_ms",
        help="Format key for fast digital clock.",
    )
    parser.set_defaults(
        show_analog=True,
        show_digital_system=True,
        show_digital_internet=True,
        show_stopwatch=True,
        show_offset=True,
    )
    args = parser.parse_args()

    key_mappings = load_key_mappings()

    PIXEL_BUFFER_SCALE = args.initial_pixel_buffer_scale
    TEXT_FIELD_SCALE = args.initial_text_field_scale

    # Base dimensions for 1.0 scale
    BASE_FB_ROWS = 756  # Reduced to make space for text overlays
    BASE_FB_COLS = 1024
    CHAR_ROWS = 30   # Set your desired ASCII grid height
    CHAR_COLS = 100  # Set your desired ASCII grid width
    init_colorama_for_windows()

    # Load configurations from JSON files
    # These will serve as templates; specific instances can override
    slow_analog_config = load_config_from_json("analog_config.json", "slow_analog")
    fast_analog_config = load_config_from_json(
        "analog_config.json", "fast_analog"
    )  # Start with base analog
    slow_digital_config = load_config_from_json("digital_config.json", "slow_digital")
    fast_digital_config = load_config_from_json(
        "digital_config.json", "fast_digital"
    )  # Start with base digital

    # Consolidate configs for easier access later if needed
    clock_configs = {
        "slow_analog": slow_analog_config,
        "fast_analog": fast_analog_config,
        "slow_digital": slow_digital_config,
        "fast_digital": fast_digital_config,
    }

    if args.configure:
        # ThemeManager needs to be initialized for backdrop cycling in config mode
        temp_theme_manager = ThemeManager(
            presets_path=os.path.join(
                os.path.dirname(__file__), "timesync", "presets", "default_themes.json"
            )
        )
        if args.backdrops:
            temp_theme_manager.current_theme.current_backdrop_path = args.backdrops[0]

        interactive_configure_mode(
            args.configure,
            clock_configs[args.configure],
            temp_theme_manager,
            key_mappings,
        )
        return  # Exit after configuration mode

    init_colorama_for_windows()
    sync_offset()
    start = time.perf_counter()

    # Initialize ThemeManager and set current theme from args
    presets_file_path = os.path.join(
        os.path.dirname(__file__), "timesync", "presets", "default_themes.json"
    )
    # Simplified path finding, assuming ThemeManager's default is usually correct
    # or the user runs from a location where `timesync/timesync/presets` is valid.

    theme_manager = ThemeManager(presets_path=presets_file_path)
    if args.backdrops:
        theme_manager.current_theme.current_backdrop_path = args.backdrops[0]

    render_backend = RenderingBackend(theme_manager)

    theme_manager.set_palette(args.theme)
    theme_manager.set_effects(args.effects)
    theme_manager.set_post_processing(args.post_processing)
    # Set initial ASCII style from args, then ThemeManager can cycle it
    theme_manager.current_theme.ascii_style = args.ascii_style

    # Store current unit set and format keys, to be cycled by new controls
    current_slow_analog_units_key = args.slow_analog_units
    current_fast_analog_units_key = args.fast_analog_units
    current_slow_digital_units_key = args.slow_digital_units
    current_fast_digital_units_key = args.fast_digital_units
    current_slow_digital_format_key = args.slow_digital_format
    current_fast_digital_format_key = args.fast_digital_format

    # Helper to get a list of TimeUnit objects for a given set name
    def get_time_units_for_set(set_name: str) -> list:
        all_defined_units = theme_manager.get_time_unit_definitions()
        unit_names_in_set = theme_manager.presets.get("time_unit_sets", {}).get(
            set_name, []
        )
        units = [
            all_defined_units[name]
            for name in unit_names_in_set
            if name in all_defined_units
        ]
        if len(units) != len(unit_names_in_set):
            print(
                f"Warning: Some units in set '{set_name}' were not found in definitions."
            )
        return units[:5]  # Max 5 units

    # Helper to get a format string for a given key
    def get_format_string_for_key(key_name: str) -> str:
        return theme_manager.presets.get("digital_format_strings", {}).get(
            key_name, "{value}"
        )

    # Initial unit sets and format strings will be fetched inside compose_full_frame

    # Initialize clock parameters by merging defaults with command-line args
    # These are base parameters, specific clock instances might have their own loaded configs
    base_analog_params = load_config_from_json("analog_config.json")
    if args.analog_clock_rect:
        # Store normalized rect instead of a pixel-based rect
        base_analog_params["clock_drawing_norm_rect"] = args.analog_clock_rect

    base_digital_params = load_config_from_json("digital_config.json")
    # Backdrop for digital_params is now handled by theme_manager.current_theme.current_backdrop_path

    # --- FRAMEBUFFER COMPOSITION ---
    def get_scaled_fb_dims():
        """Return framebuffer dimensions scaled by ``PIXEL_BUFFER_SCALE``."""
        return int(BASE_FB_ROWS * PIXEL_BUFFER_SCALE), int(
            BASE_FB_COLS * PIXEL_BUFFER_SCALE
        )

    def get_char_cell_dims(char_rows = CHAR_ROWS, char_cols = CHAR_COLS) -> tuple[int, int]:
        """
        Dynamically compute the pixel size of each character cell so that the framebuffer
        is divided evenly into a fixed number of character rows and columns.
        """
        fb_rows, fb_cols = get_scaled_fb_dims()
        cell_h = max(1, fb_rows // char_rows)
        cell_w = max(1, fb_cols // char_cols)
        return cell_h, cell_w

    fb_rows, fb_cols = get_scaled_fb_dims()
    cell_h, cell_w = get_char_cell_dims(CHAR_ROWS, CHAR_COLS)

    framebuffer = PixelFrameBuffer(
        (fb_rows, fb_cols), diff_threshold=20
    )  # Initialized with scaled dims
    stop_event = threading.Event()

    # Display state flags
    display_state = {
        "analog_clocks": args.show_analog,
        "slow_digital_clock": args.show_digital_system,
        "fast_digital_clock": args.show_digital_internet,
        "stopwatch": args.show_stopwatch,
        "offset_info": args.show_offset,
        "legend": True,  # Show legend by default
    }

    def compose_full_frame(
        system_time_obj, internet_time_obj, stopwatch_td_obj, offset_val
    ):
        current_fb_rows, current_fb_cols = get_scaled_fb_dims()
        # Initialize buffer with black pixels
        buf = np.full(
            (current_fb_rows, current_fb_cols, 3), [0, 0, 0], dtype=np.uint8
        )  # This buffer is ONLY for the clocks now
        row = 0
        available_width = current_fb_cols  # Pixel width
        spacer_color = [5, 5, 10]  # Dark spacer
        spacer_height = 1

        # Determine time objects for slow/fast clocks
        slow_time_obj = internet_time_obj  # Or system_time_obj
        fast_time_obj = stopwatch_td_obj

        current_backdrop_path = theme_manager.current_theme.current_backdrop_path
        # Fetch current unit sets and formats based on keys
        slow_analog_units = get_time_units_for_set(current_slow_analog_units_key)
        fast_analog_units = get_time_units_for_set(current_fast_analog_units_key)
        slow_digital_units = get_time_units_for_set(current_slow_digital_units_key)
        fast_digital_units = get_time_units_for_set(current_fast_digital_units_key)
        slow_digital_format = get_format_string_for_key(current_slow_digital_format_key)
        fast_digital_format = get_format_string_for_key(current_fast_digital_format_key)

        # Unified clock rendering using ascii_digits.generate_clock
        # Render analog face as an image, then convert to raw pixels
        img = ClockRenderer.render_analog(
            time_obj=system_time_obj,
            units=slow_analog_units,
            bounding_size=(current_fb_cols, current_fb_rows),
            backdrop_image_path=current_backdrop_path,
            theme_manager=theme_manager,
            render_backend=render_backend,
            **base_analog_params
        )
        unified_clock = np.array(img.convert("RGB"))

        if unified_clock is not None:
            h_arr, w_arr, _ = unified_clock.shape
            crop_h = min(h_arr, current_fb_rows)
            crop_w = min(w_arr, current_fb_cols)
            buf[:crop_h, :crop_w] = unified_clock[:crop_h, :crop_w]

        return buf

    def render_fn(framebuffer_ref):  # Pass framebuffer to allow reinitialization
        elapsed = time.perf_counter() - start
        offset = get_offset()
        system = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
        internet = system + _dt.timedelta(seconds=offset)

        h, rem = divmod(int(elapsed * 1000), 3600 * 1000)
        m, rem = divmod(rem, 60 * 1000)
        s, ms = divmod(rem, 1000)
        stopwatch_td = _dt.timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)

        # Check if framebuffer dimensions need to change due to scale factors
        new_fb_rows, new_fb_cols = get_scaled_fb_dims()
        if (
            framebuffer_ref.buffer_shape[0] != new_fb_rows
            or framebuffer_ref.buffer_shape[1] != new_fb_cols
        ):
            framebuffer_ref.resize((new_fb_rows, new_fb_cols))

        frame = compose_full_frame(system, internet, stopwatch_td, offset)
        img = Image.fromarray(frame, mode="RGB")
        img = render_backend.process(img)
        if img.mode != "RGB":
            # RenderingBackend may produce an RGBA image when effects use
            # transparency. Convert back to RGB so PixelFrameBuffer receives
            # the expected three-channel data.
            img = img.convert("RGB")
        return np.array(img)

    render_thread = threading.Thread(
        target=render_loop,
        args=(
            framebuffer,
            lambda: render_fn(framebuffer),
            10,
            stop_event,
        ),  # Pass framebuffer to render_fn
        daemon=True,
    )
    render_thread.start()

    # Start the input thread AFTER the render thread
    full_clear_and_reset_cursor()
    input_buffer = ""
    input_queue = queue.Queue()
    input_thread = threading.Thread(
        target=input_thread_fn, args=(input_queue, stop_event), daemon=True
    )
    input_thread.start()

    # text_overlay_start_row tracks where text drawing begins relative to the

    # For periodic keyframe redraw
    KEYFRAME_INTERVAL_SECONDS = 30  # Redraw everything every 30 seconds
    last_keyframe_time = time.perf_counter()

    def quit():
        stop_event.set()

    def toggle_legend():
        display_state["legend"] = not display_state["legend"]

    def cycle_ascii_style_forward():
        theme_manager.cycle_ascii_style()

    def cycle_ascii_style_backward():
        theme_manager.cycle_ascii_style(-1)

    def cycle_palette_forward():
        theme_manager.cycle_palette(1)

    def cycle_palette_backward():
        theme_manager.cycle_palette(-1)

    def cycle_effects_forward():
        theme_manager.cycle_effects_preset(1)

    def cycle_effects_backward():
        theme_manager.cycle_effects_preset(-1)

    def cycle_post_processing_forward():
        theme_manager.cycle_post_processing_preset(1)

    def cycle_post_processing_backward():
        theme_manager.cycle_post_processing_preset(-1)

    def toggle_clock_inversion():
        theme_manager.toggle_clock_inversion()

    def toggle_backdrop_inversion():
        theme_manager.toggle_backdrop_inversion()

    def cycle_backdrop_forward():
        theme_manager.cycle_backdrop(args.backdrops or [], 1)

    def cycle_backdrop_backward():
        theme_manager.cycle_backdrop(args.backdrops or [], -1)

    def enter_config_mode():
        # Example: just print for now
        print("Config mode entered (implement mode stack as needed)")

    def cycle_config_focus():
        print("Cycle config focus (implement focus logic as needed)")

    # --- Config mode handlers for analog ---
    def config_quit():
        print("Config quit (implement config quit logic)")

    def config_save():
        print("Config save (implement config save logic)")

    def config_left():
        # Move analog clock rect left
        for clock in ["slow_analog", "fast_analog"]:
            rect = list(clock_configs[clock].get("clock_drawing_rect_on_canvas", [20, 20, 80, 80]))
            rect[0] -= 2
            clock_configs[clock]["clock_drawing_rect_on_canvas"] = tuple(max(0, val) for val in rect)
        print("Analog clock rect moved left")

    def config_right():
        for clock in ["slow_analog", "fast_analog"]:
            rect = list(clock_configs[clock].get("clock_drawing_rect_on_canvas", [20, 20, 80, 80]))
            rect[0] += 2
            clock_configs[clock]["clock_drawing_rect_on_canvas"] = tuple(max(0, val) for val in rect)
        print("Analog clock rect moved right")

    def config_up():
        for clock in ["slow_analog", "fast_analog"]:
            rect = list(clock_configs[clock].get("clock_drawing_rect_on_canvas", [20, 20, 80, 80]))
            rect[1] -= 2
            clock_configs[clock]["clock_drawing_rect_on_canvas"] = tuple(max(0, val) for val in rect)
        print("Analog clock rect moved up")

    def config_down():
        for clock in ["slow_analog", "fast_analog"]:
            rect = list(clock_configs[clock].get("clock_drawing_rect_on_canvas", [20, 20, 80, 80]))
            rect[1] += 2
            clock_configs[clock]["clock_drawing_rect_on_canvas"] = tuple(max(0, val) for val in rect)
        print("Analog clock rect moved down")

    def config_width_increase():
        for clock in ["slow_analog", "fast_analog"]:
            rect = list(clock_configs[clock].get("clock_drawing_rect_on_canvas", [20, 20, 80, 80]))
            rect[2] += 2
            clock_configs[clock]["clock_drawing_rect_on_canvas"] = tuple(max(0, val) for val in rect)
        print("Analog clock width increased")

    def config_width_decrease():
        for clock in ["slow_analog", "fast_analog"]:
            rect = list(clock_configs[clock].get("clock_drawing_rect_on_canvas", [20, 20, 80, 80]))
            rect[2] = max(2, rect[2] - 2)
            clock_configs[clock]["clock_drawing_rect_on_canvas"] = tuple(max(0, val) for val in rect)
        print("Analog clock width decreased")

    def config_height_increase():
        for clock in ["slow_analog", "fast_analog"]:
            rect = list(clock_configs[clock].get("clock_drawing_rect_on_canvas", [20, 20, 80, 80]))
            rect[3] += 2
            clock_configs[clock]["clock_drawing_rect_on_canvas"] = tuple(max(0, val) for val in rect)
        print("Analog clock height increased")

    def config_height_decrease():
        for clock in ["slow_analog", "fast_analog"]:
            rect = list(clock_configs[clock].get("clock_drawing_rect_on_canvas", [20, 20, 80, 80]))
            rect[3] = max(2, rect[3] - 2)
            clock_configs[clock]["clock_drawing_rect_on_canvas"] = tuple(max(0, val) for val in rect)
        print("Analog clock height decreased")

    def config_ascii_diameter_increase():
        for clock in ["slow_analog", "fast_analog"]:
            clock_configs[clock]["target_ascii_diameter"] = clock_configs[clock].get("target_ascii_diameter", 22) + 1
        print("Analog ASCII diameter increased")

    def config_ascii_diameter_decrease():
        for clock in ["slow_analog", "fast_analog"]:
            clock_configs[clock]["target_ascii_diameter"] = max(5, clock_configs[clock].get("target_ascii_diameter", 22) - 1)
        print("Analog ASCII diameter decreased")

    def config_canvas_increase():
        for clock in ["slow_analog", "fast_analog"]:
            clock_configs[clock]["canvas_size_px"] = clock_configs[clock].get("canvas_size_px", 120) + 10
        print("Analog canvas size increased")

    def config_canvas_decrease():
        for clock in ["slow_analog", "fast_analog"]:
            clock_configs[clock]["canvas_size_px"] = max(20, clock_configs[clock].get("canvas_size_px", 120) - 10)
        print("Analog canvas size decreased")

    # --- Config mode handlers for digital ---
    def config_font_increase():
        for clock in ["slow_digital", "fast_digital"]:
            clock_configs[clock]["font_size"] = clock_configs[clock].get("font_size", 40) + 2
        print("Digital font size increased")

    def config_font_decrease():
        for clock in ["slow_digital", "fast_digital"]:
            clock_configs[clock]["font_size"] = max(8, clock_configs[clock].get("font_size", 40) - 2)
        print("Digital font size decreased")

    def config_left():
        for clock in ["slow_digital", "fast_digital"]:
            clock_configs[clock]["target_ascii_width"] = max(10, clock_configs[clock].get("target_ascii_width", 60) - 2)
        print("Digital ASCII width decreased")

    def config_right():
        for clock in ["slow_digital", "fast_digital"]:
            clock_configs[clock]["target_ascii_width"] = clock_configs[clock].get("target_ascii_width", 60) + 2
        print("Digital ASCII width increased")

    def config_up():
        for clock in ["slow_digital", "fast_digital"]:
            clock_configs[clock]["target_ascii_height"] = max(3, clock_configs[clock].get("target_ascii_height", 7) - 1)
        print("Digital ASCII height decreased")

    def config_down():
        for clock in ["slow_digital", "fast_digital"]:
            clock_configs[clock]["target_ascii_height"] = clock_configs[clock].get("target_ascii_height", 7) + 1
        print("Digital ASCII height increased")


    action_handlers = {
        "normal": {
            "quit": quit,
            "toggle_legend": toggle_legend,
            "cycle_ascii_style_forward": cycle_ascii_style_forward,
            "cycle_ascii_style_backward": cycle_ascii_style_backward,
            "cycle_palette_forward": cycle_palette_forward,
            "cycle_palette_backward": cycle_palette_backward,
            "cycle_effects_forward": cycle_effects_forward,
            "cycle_effects_backward": cycle_effects_backward,
            "cycle_post_processing_forward": cycle_post_processing_forward,
            "cycle_post_processing_backward": cycle_post_processing_backward,
            "toggle_clock_inversion": toggle_clock_inversion,
            "toggle_backdrop_inversion": toggle_backdrop_inversion,
            "cycle_backdrop_forward": cycle_backdrop_forward,
            "cycle_backdrop_backward": cycle_backdrop_backward,
            "enter_config_mode": enter_config_mode,
            "cycle_config_focus": cycle_config_focus,
        },
        "config": {
            "analog": {
                "config_quit": config_quit,
                "config_save": config_save,
                "config_left": config_left,
                "config_right": config_right,
                "config_up": config_up,
                "config_down": config_down,
                "config_width_increase": config_width_increase,
                "config_width_decrease": config_width_decrease,
                "config_height_increase": config_height_increase,
                "config_height_decrease": config_height_decrease,
                "config_ascii_diameter_increase": config_ascii_diameter_increase,
                "config_ascii_diameter_decrease": config_ascii_diameter_decrease,
                "config_canvas_increase": config_canvas_increase,
                "config_canvas_decrease": config_canvas_decrease,
            },
            "digital": {
                "config_quit": config_quit,
                "config_save": config_save,
                "config_left": config_left,
                "config_right": config_right,
                "config_up": config_up,
                "config_down": config_down,
                "config_font_increase": config_font_increase,
                "config_font_decrease": config_font_decrease,
            }
        }
    }

    # ----------------------------------------------------------------------
    # Generic stub for any missing handler functions
    def missing_handler_stub(*args, **kwargs):
        print("Warning: Called a missing handler stub. Please implement this action.")
    # ----------------------------------------------------------------------

    # Example: If you add a new action but forget to define it, do:
    # action_handlers["normal"]["new_action"] = missing_handler_stub
    # or for config:
    # action_handlers["config"]["analog"]["new_config_action"] = missing_handler_stub

    menu_resolver = MenuResolver(key_mappings, action_handlers)
    input_dispatcher = InputDispatcher(menu_resolver)

    try:
        while True:
            # --- Drawing Phase ---
            # The render_thread updates the framebuffer's render buffer.
            # The main loop gets the diff from the framebuffer and draws it.

            # Get changed pixels from the framebuffer
            diff_pixels = framebuffer.get_diff_and_promote()
            cell_h, cell_w = get_char_cell_dims()
            unique_cells: dict[tuple[int, int], np.ndarray] = {}

            for y, x, color in diff_pixels:
                # Determine the top-left pixel coordinates of the character cell this pixel belongs to
                cell_y_start_pixel = (y // cell_h) * cell_h
                cell_x_start_pixel = (x // cell_w) * cell_w

                if (cell_y_start_pixel, cell_x_start_pixel) not in unique_cells:
                    # Extract the subunit using the target cell dimensions.
                    # NumPy slicing will handle edges by returning a smaller array.
                    sub_slice = framebuffer.buffer_display[
                        cell_y_start_pixel : cell_y_start_pixel + cell_h,
                        cell_x_start_pixel : cell_x_start_pixel + cell_w,
                    ]

                    # Ensure 'sub_slice' is padded to (cell_h, cell_w, 3) if it's smaller.
                    # This typically happens for subunits at the right or bottom edges.
                    if sub_slice.shape[0] != cell_h or sub_slice.shape[1] != cell_w:
                        padded_sub = np.full((cell_h, cell_w, 3), [0, 0, 0], dtype=np.uint8)  # Default to black
                        padded_sub[:sub_slice.shape[0], :sub_slice.shape[1], :] = sub_slice
                        sub_to_store = padded_sub
                    else:
                        sub_to_store = sub_slice
                    unique_cells[(cell_y_start_pixel, cell_x_start_pixel)] = sub_to_store

            changed = [(cy, cx, data) for (cy, cx), data in unique_cells.items()]
            # Draw the changed pixels (the clocks)
            # If diff_pixels is empty but a redraw was forced, 'changed' will also be empty.
            # The draw_diff will still be called, but it won't do much if 'changed' is empty.
            # The key is that get_diff_and_promote() would have returned all pixels if forced.
            # However, the current draw_diff takes 'changed_subunits' which are already diffed.
            # The PixelFrameBuffer.get_diff_and_promote() now returns all pixels, and thus `changed` will too.
            # We need to pass the correct char_cell_pixel_height/width to draw_diff
            # These should correspond to the pixel dimensions of a single character cell
            # as used by the rendering functions (print_analog_clock, print_digital_clock).
            # This is complex because different elements might use different effective pixel sizes per char.
            # In practice ``get_char_cell_dims`` derives the pixel size of a
            # character cell based on ``PIXEL_BUFFER_SCALE``.  Avoid assuming
            # a hard 1:1 mapping so diff calculations continue to work on
            # sub-character pixel regions.
            cell_h, cell_w = get_char_cell_dims()
            draw_diff(
                changed,
                char_cell_pixel_height=cell_h,
                char_cell_pixel_width=cell_w,
                active_ascii_ramp=theme_manager.get_current_ascii_ramp(),
            )

            # --- Text Overlay Drawing Phase ---
            # Draw text elements directly as overlays below the pixel buffer
            # The main ASCII display (from draw_diff) occupies CHAR_ROWS lines, starting at base_row=1.
            # So, text overlays should start at character row CHAR_ROWS + 1.
            text_overlay_actual_start_row = CHAR_ROWS + 1 # Assumes draw_diff base_row=1
            current_text_overlay_row = text_overlay_actual_start_row

            # Use CHAR_COLS for the width of the overlay area to align with the main display
            overlay_available_cols = CHAR_COLS
            # Clear the text overlay area before drawing new text
            # This prevents old text from lingering if new text is shorter
            text_overlay_height = (
                BASE_FB_ROWS - BASE_FB_ROWS
            )  # Calculate total height needed for overlays
            # This calculation is tricky as text height depends on content and scale.
            # A simpler approach is to clear a fixed number of lines at the bottom.
            # Let's clear enough lines for Stopwatch (1), Offset (1), Legend (approx 10 lines).
            # Total ~12 lines. BASE_FB_ROWS is 35, so 45 total. Need ~10 lines below buffer.
            # Let's clear from text_overlay_start_row to the end of the terminal.
            # This requires knowing terminal height, which is hard.
            # Alternative: Clear a fixed number of lines based on max possible overlays.
            # Stopwatch (1) + Offset (1) + Legend (approx 10) = ~12 lines.
            # Let's clear 15 lines from text_overlay_start_row.
            clear_lines_count = 15
            for r in range(
                current_text_overlay_row, current_text_overlay_row + clear_lines_count
            ):
                draw_text_overlay(
                    r, 1, " " * overlay_available_cols, Style.RESET_ALL
                )  # Clear line with spaces

            # Get current time objects for text overlays
            elapsed = time.perf_counter() - start
            offset = get_offset()
            system = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
            internet = system + _dt.timedelta(seconds=offset)
            h, rem = divmod(int(elapsed * 1000), 3600 * 1000)
            m, rem = divmod(rem, 60 * 1000)
            s, ms = divmod(rem, 1000)
            stopwatch_td = _dt.timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)

            if display_state["stopwatch"]:
                stopwatch_text = f"Stopwatch: {stopwatch_td.total_seconds():09.3f}"
                draw_text_overlay(
                    current_text_overlay_row,
                    1,
                    stopwatch_text[:overlay_available_cols], # Truncate if necessary
                    Style.BRIGHT + Fore.YELLOW,
                )  # Example color
                current_text_overlay_row += 1  # Move to next line

            if display_state["offset_info"]:
                offset_text = f"Offset: {_dt.timedelta(seconds=offset)}"
                draw_text_overlay(
                    current_text_overlay_row, 1, offset_text[:overlay_available_cols], Style.BRIGHT + Fore.CYAN
                )
                current_text_overlay_row += 1  # Move to next line

            if display_state["legend"]:
                legend_start_col = 1
                legend_col_width = overlay_available_cols // 2  # Split legend into two columns

                def collect_legend_parts(mapping):
                    legend = []
                    if isinstance(mapping, dict):
                        for v in mapping.values():
                            if isinstance(v, dict) and "keys" in v:
                                keys_str = "/".join(v["keys"])
                                desc = v.get("description", "")
                                legend.append(f"{keys_str}: {desc}")
                            elif isinstance(v, dict):
                                legend.extend(collect_legend_parts(v))
                    return legend

                mode = menu_resolver.mode_stack[-1] if menu_resolver.mode_stack else "normal"
                focused_clock_type = getattr(menu_resolver, "focused_clock_type", None)

                if mode == "normal":
                    legend_mapping = key_mappings.get("normal", {})
                elif mode == "config" and focused_clock_type:
                    legend_mapping = key_mappings.get("config", {}).get(focused_clock_type, {})
                else:
                    legend_mapping = {}

                legend_parts = collect_legend_parts(legend_mapping)

                num_legend_items = len(legend_parts)
                col1_items = legend_parts[: (num_legend_items + 1) // 2]
                col2_items = legend_parts[(num_legend_items + 1) // 2 :]

                for i in range(max(len(col1_items), len(col2_items))):
                    if (
                        current_text_overlay_row
                        >= text_overlay_actual_start_row + clear_lines_count
                    ):  # Prevent drawing outside cleared area
                        break
                    line_text_parts = []
                    ljust_val = max(10, overlay_available_cols // 2 - 2) # Adjust ljust based on available width
                    if i < len(col1_items):
                        line_text_parts.append(col1_items[i].ljust(ljust_val))
                    if i < len(col2_items):
                        line_text_parts.append(col2_items[i])
                    legend_line_text = "  ".join(line_text_parts)
                    draw_text_overlay(
                        current_text_overlay_row,
                        legend_start_col,
                        legend_line_text[:overlay_available_cols], # Truncate legend line
                        Style.RESET_ALL,
                    )
                    current_text_overlay_row += 1

            # Drain the input queue
            while not input_queue.empty():
                key = input_queue.get()
                input_buffer += key

            action_processed_this_loop = input_dispatcher.dispatch(input_buffer)
            input_buffer = ""  # Clear after dispatch

            # Trigger keyframe if an action was processed
            if action_processed_this_loop:
                framebuffer.force_full_redraw_next_frame()
                # Also force redraw of text overlays? Not needed as they are redrawn every frame.
            # Periodic keyframe trigger
            current_loop_time = time.perf_counter()
            if current_loop_time - last_keyframe_time >= KEYFRAME_INTERVAL_SECONDS:
                framebuffer.force_full_redraw_next_frame()
                last_keyframe_time = current_loop_time

            if stop_event.is_set():  # Check if we should break out of the main loop
                break
            time.sleep(args.refresh_rate)

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        input_thread.join(timeout=2.0)
        render_thread.join(timeout=2.0)
        full_clear_and_reset_cursor()
        print("Demo stopped.")
        final_offset = get_offset()
        final_system = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
        final_internet = final_system + _dt.timedelta(seconds=final_offset)
        print(f"Final system time: {final_system.strftime('%H:%M:%S')}")
        print(f"Final internet time: {final_internet.strftime('%H:%M:%S')}")
        print(f"Offset: {_dt.timedelta(seconds=final_offset)}")


if __name__ == "__main__":
    main()
