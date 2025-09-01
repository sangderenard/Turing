# Pixel Art Classifier Task

This learning task emits simple 8×8 pixel art shapes with configurable noise.
Models receive a noisy version of one of the shapes and must classify which
shape was chosen. Available shapes include a square, triangle, X and diamond.

Two noise modes are available:

* **spectral** – (default) adds noise that shares the magnitude spectrum of the
  clean shape but uses randomized phase.
* **corrupt** – randomly replaces a fraction of the shape's pixels with uniform
  noise, simulating recognition from corrupted inputs.

Samples produced by `pump_queue` are tuples of `(input, target, category)`
where:

* `input` – noisy shape array of shape `(1, 8, 8)`.
* `target` – the clean shape image (unused by the loss composer).
* `category` – dictionary with `label` and `name` entries for the shape.

The loss composer exposes `NUM_LOGITS` equal to the number of shapes and
returns a cross‑entropy loss over the logits slice.
