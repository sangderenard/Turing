"""Image viewer that dumps an image to the ASCII renderer with ramp and post-processing preset."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image

from .draw import get_changed_subunits, draw_diff, default_subunit_batch_to_chars
from .console import full_clear_and_reset_cursor, reset_cursor_to_top
from .theme_manager import ThemeManager

# Default settings
DEFAULT_IMAGE_PATH = Path(__file__).with_name("analogback.png")
DEFAULT_CHAR_H = 32
DEFAULT_CHAR_W = 16
DEFAULT_RAMP_STYLE = "block"  # Can be changed to any ramp in ascii_styles
DEFAULT_POST_PROCESSING = "high_contrast"  # Can be changed to any preset in post_processing


def load_pixels(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)




def menu():
    theme_manager = ThemeManager()
    char_h = DEFAULT_CHAR_H
    char_w = DEFAULT_CHAR_W
    ramp_style = DEFAULT_RAMP_STYLE
    post_processing = DEFAULT_POST_PROCESSING
    image_path = str(DEFAULT_IMAGE_PATH)

    show_bg = True
    show_fg = True

    fps = -1  # -1 means as fast as possible
    def print_menu():
        print("\n--- ASCII Image Viewer Menu ---")
        print(f"1. Char Height (H): {char_h}")
        print(f"2. Char Width (W): {char_w}")
        print(f"3. Style (Ramp): {ramp_style}")
        print(f"4. Post-Processing: {post_processing}")
        print(f"5. Image File: {image_path}")
        print(f"6. Show Background Color: {'ON' if show_bg else 'OFF'}")
        print(f"7. Show Foreground Color: {'ON' if show_fg else 'OFF'}")
        print(f"8. Render Image/Animation")
        print(f"10. Set Animation FPS (current: {'max' if fps == -1 else fps})")
        print("9. Exit")

    while True:
        print_menu()
        choice = input("Select option (1-9,10): ").strip()
        if choice == "1":
            try:
                char_h = int(input("Enter new char height: ").strip())
            except Exception:
                print("Invalid input.")
        elif choice == "2":
            try:
                char_w = int(input("Enter new char width: ").strip())
            except Exception:
                print("Invalid input.")
        elif choice == "3":
            ramps = list(theme_manager.presets.get("ascii_styles", {}).keys())
            print("Available styles:", ", ".join(ramps))
            val = input(f"Enter style (current: {ramp_style}): ").strip()
            if val in ramps:
                ramp_style = val
            else:
                print("Invalid style.")
        elif choice == "4":
            pps = list(theme_manager.presets.get("post_processing", {}).keys())
            print("Available post-processing:", ", ".join(pps))
            val = input(f"Enter post-processing (current: {post_processing}): ").strip()
            if val in pps:
                post_processing = val
            else:
                print("Invalid post-processing preset.")
        elif choice == "5":
            val = input(f"Enter image file path (current: {image_path}): ").strip()
            if val:
                image_path = val
        elif choice == "6":
            show_bg = not show_bg
            print(f"Show Background Color set to {'ON' if show_bg else 'OFF'}.")
        elif choice == "7":
            show_fg = not show_fg
            print(f"Show Foreground Color set to {'ON' if show_fg else 'OFF'}.")
        elif choice == "8":
            import time
            while True:
                try:
                    theme_manager.current_theme.ascii_style = ramp_style
                    theme_manager.set_post_processing(post_processing)
                    img = Image.open(image_path).convert("RGB")
                    img = theme_manager.apply_theme(img)
                    frame = np.array(img)
                    h, w, c = frame.shape
                    # 1. Clear with all white (no terminal clear, just diff)
                    old_frame = np.zeros_like(frame)
                    white = np.ones_like(frame) * 255
                    changed = get_changed_subunits(old_frame, white, char_h, char_w)
                    draw_diff(
                        changed,
                        char_cell_pixel_height=char_h,
                        char_cell_pixel_width=char_w,
                        subunit_to_char_kernel=default_subunit_batch_to_chars,
                        active_ascii_ramp=theme_manager.get_current_ascii_ramp(),
                        enable_fg_color=show_fg,
                        enable_bg_color=show_bg,
                    )
                    sys.stdout.flush()
                    # Minimal delay for speed
                    time.sleep(0.01)
                    # 2. Clear with all black
                    old_frame = white.copy()
                    black = np.zeros_like(frame)
                    changed = get_changed_subunits(old_frame, black, char_h, char_w)
                    draw_diff(
                        changed,
                        char_cell_pixel_height=char_h,
                        char_cell_pixel_width=char_w,
                        subunit_to_char_kernel=default_subunit_batch_to_chars,
                        active_ascii_ramp=theme_manager.get_current_ascii_ramp(),
                        enable_fg_color=show_fg,
                        enable_bg_color=show_bg,
                    )
                    sys.stdout.flush()
                    time.sleep(0.01)
                    # 3. Animate image
                    def roll_frame(f, shift_y=0, shift_x=0):
                        return np.roll(np.roll(f, shift_y, axis=0), shift_x, axis=1)
                    def animate(frames, mode_name):
                        for idx, (oldf, newf) in enumerate(frames):
                            changed = get_changed_subunits(oldf, newf, char_h, char_w)
                            draw_diff(
                                changed,
                                char_cell_pixel_height=char_h,
                                char_cell_pixel_width=char_w,
                                subunit_to_char_kernel=default_subunit_batch_to_chars,
                                active_ascii_ramp=theme_manager.get_current_ascii_ramp(),
                                enable_fg_color=show_fg,
                                enable_bg_color=show_bg,
                            )
                            #sys.stdout.flush()
                            if fps > 0:
                                time.sleep(1.0 / fps)
                    # Vertical roll (quarter stride of char_h)
                    frames = []
                    v_stride = max(1, char_h // 4)
                    for i in range(0, h, v_stride):
                        frames.append((frame if i == 0 else roll_frame(frame, i-v_stride, 0), roll_frame(frame, i, 0)))
                    animate(frames, "vertical")
                    # Horizontal roll (quarter stride of char_w)
                    frames = []
                    h_stride = max(1, char_w // 4)
                    for i in range(0, w, h_stride):
                        frames.append((frame if i == 0 else roll_frame(frame, 0, i-h_stride), roll_frame(frame, 0, i)))
                    animate(frames, "horizontal")
                    # Sinusoidal roll
                    frames = []
                    for t in range(max(h, w)):
                        y_shift = int((np.sin(2 * np.pi * t / h) * h / 4))
                        x_shift = int((np.sin(2 * np.pi * t / w + np.pi/2) * w / 4))
                        prev_y = int((np.sin(2 * np.pi * (t-1) / h) * h / 4)) if t > 0 else 0
                        prev_x = int((np.sin(2 * np.pi * (t-1) / w + np.pi/2) * w / 4)) if t > 0 else 0
                        frames.append((roll_frame(frame, prev_y, prev_x), roll_frame(frame, y_shift, x_shift)))
                    animate(frames, "sinusoidal")
                    # Stationary hold
                    changed = get_changed_subunits(frames[-1][1], frame, char_h, char_w)
                    draw_diff(
                        changed,
                        char_cell_pixel_height=char_h,
                        char_cell_pixel_width=char_w,
                        subunit_to_char_kernel=default_subunit_batch_to_chars,
                        active_ascii_ramp=theme_manager.get_current_ascii_ramp(),
                        enable_fg_color=show_fg,
                        enable_bg_color=show_bg,
                    )
                    sys.stdout.flush()
                    # Prompt for repeat or continue
                    user_input = input("\nPress Enter to continue, or type 'repeat' to play again: ").strip().lower()
                    if user_input == "repeat":
                        continue  # Replay animation
                    reset_cursor_to_top()
                    break  # Exit animation loop
                except Exception as e:
                    print(f"Error rendering image: {e}")
                    break
        elif choice == "10":
            try:
                val = input("Enter FPS for animation (-1 for max speed): ").strip()
                fps = int(val)
            except Exception:
                print("Invalid input.")
        elif choice == "9":
            print("Exiting.")
            break
        else:
            print("Invalid option.")



if __name__ == "__main__":
    menu()
