"""Image viewer that dumps an image to the ASCII renderer with ramp and post-processing preset."""
from pathlib import Path
import numpy as np
from PIL import Image

from .console import reset_cursor_to_top
from .theme_manager import ThemeManager
from ..render_chooser import RenderChooser

# Default settings
DEFAULT_IMAGE_PATH = Path(__file__).with_name("analogback.png")
DEFAULT_CHAR_H = 32
DEFAULT_CHAR_W = 16
DEFAULT_RAMP_STYLE = "block"  # Can be changed to any ramp in ascii_styles
DEFAULT_POST_PROCESSING = "high_contrast"  # Can be changed to any preset in post_processing


def load_pixels(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _downsample(frame: np.ndarray, char_h: int, char_w: int) -> np.ndarray:
    """Reduce ``frame`` to character-cell resolution.

    The ``RenderChooser`` and underlying :class:`AsciiRenderer` operate on a
    single pixel per terminal cell.  ``ascii_diff`` historically allowed each
    character to represent a block of pixels.  To retain this behaviour we
    average blocks of ``char_h``×``char_w`` pixels down to one value.
    """

    if char_h <= 1 and char_w <= 1:
        return frame
    h, w, c = frame.shape
    new_h = h // char_h
    new_w = w // char_w
    if new_h == 0 or new_w == 0:
        return frame
    cropped = frame[: new_h * char_h, : new_w * char_w]
    reshaped = cropped.reshape(new_h, char_h, new_w, char_w, c)
    reduced = reshaped.mean(axis=(1, 3)).astype(np.uint8)
    return reduced




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
            theme_manager.current_theme.ascii_style = ramp_style
            theme_manager.set_post_processing(post_processing)
            img = Image.open(image_path).convert("RGB")
            img = theme_manager.apply_theme(img)
            frame = _downsample(np.array(img), char_h, char_w)
            h, w, _ = frame.shape
            rc = RenderChooser(w, h, mode="ascii")
            try:
                def roll_frame(f: np.ndarray, shift_y: int = 0, shift_x: int = 0) -> np.ndarray:
                    return np.roll(np.roll(f, shift_y, axis=0), shift_x, axis=1)

                while True:
                    white = np.ones_like(frame) * 255
                    rc.render({"image": white})
                    time.sleep(0.01)
                    rc.render({"image": np.zeros_like(frame)})
                    time.sleep(0.01)

                    v_stride = max(1, h // 4)
                    for i in range(0, h, v_stride):
                        rc.render({"image": roll_frame(frame, i, 0)})
                        if fps > 0:
                            time.sleep(1.0 / fps)

                    h_stride = max(1, w // 4)
                    for i in range(0, w, h_stride):
                        rc.render({"image": roll_frame(frame, 0, i)})
                        if fps > 0:
                            time.sleep(1.0 / fps)

                    for t in range(max(h, w)):
                        y_shift = int((np.sin(2 * np.pi * t / h) * h / 4))
                        x_shift = int((np.sin(2 * np.pi * t / w + np.pi / 2) * w / 4))
                        rc.render({"image": roll_frame(frame, y_shift, x_shift)})
                        if fps > 0:
                            time.sleep(1.0 / fps)

                    rc.render({"image": frame})
                    # Ensure all pending frames have been printed before prompting the user.
                    if rc.mode == "ascii" and getattr(rc, "_ascii_queue", None) is not None:
                        rc._ascii_queue.join()
                    user_input = (
                        input("\nPress Enter to continue, or type 'repeat' to play again: ")
                        .strip()
                        .lower()
                    )
                    if user_input == "repeat":
                        continue
                    reset_cursor_to_top()
                    break
            except Exception as e:
                print(f"Error rendering image: {e}")
            finally:
                rc.close()
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
