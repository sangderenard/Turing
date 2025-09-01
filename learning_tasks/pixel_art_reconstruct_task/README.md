# Pixel Art Reconstruction Task

This task provides latent noise fields whose spectral magnitude matches one of
the embedded pixel art shapes.  The model must reconstruct the original shape
from the noise. Supported shapes are the square, triangle, X and diamond.

`pump_queue` yields `(input, target, category)` tuples:

* `input` – spectral noise shaped like a chosen pixel art image, `(1, 8, 8)`.
* `target` – the corresponding clean shape.
* `category` – dictionary with `label` and `name` describing the shape.

The loss composer supplies a mean‑squared error between the prediction and the
target image.
