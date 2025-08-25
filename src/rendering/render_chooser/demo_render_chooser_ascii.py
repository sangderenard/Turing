"""Demo for using RenderChooser in ASCII render mode with render()."""
from src.rendering.render_chooser import RenderChooser
import time

if __name__ == "__main__":
    # Set width and height to be multiples of the default tile size (char_cell_pixel_width=16, char_cell_pixel_height=32)
    # This avoids tile size mismatch at the edges for the default AsciiRenderer config
    width = 320  # 2 tiles wide (16*2)
    height = 320  # 1 tile high (32*1)
    chooser = RenderChooser(width=width, height=height, mode="ascii")
    try:
        # Demo: draw a diagonal line using the 'edges' state
        state = {
            "edges": [((0, 0), (width-1, height-1))],
        }
        print("Rendering a diagonal line with render()...")
        chooser.render(state)
        time.sleep(1)
        # Demo: draw a triangle
        state = {
            "triangles": [((2, 2), (width//2, 2), (width//3, height-2))],
        }
        print("Rendering a triangle with render()...")
        chooser.render(state)
        time.sleep(1)
        # Demo: draw points
        state = {
            "points": [(width//4, height//2), (width//2, height//2), (3*width//4, height//2)],
        }
        print("Rendering points with render()...")
        chooser.render(state)
        time.sleep(1)
    finally:
        chooser.close()
        print("Demo complete.")
