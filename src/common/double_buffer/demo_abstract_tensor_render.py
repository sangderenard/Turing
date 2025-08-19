
import time
import numpy as np
from src.common.tensors.abstraction import AbstractTensor
from src.rendering.render_chooser import RenderChooser

def make_texture_tensor(width, height):
    arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return AbstractTensor.tensor_from_list(arr.tolist())

if __name__ == "__main__":
    width, height = 16, 8
    frames = 5
    # Try different renderer modes: 'ascii', 'pygame', 'opengl' (if available)
    for mode in [None, 'ascii']:
        print(f"\n--- RenderChooser mode: {mode or 'default'} ---")
        chooser = RenderChooser(width, height, mode=mode)
        for frame in range(frames):
            tex = make_texture_tensor(width, height)
            state = {"textures": [(tex, 0, 0)]}
            chooser.render(state)
            time.sleep(0.2)
        chooser.close()
