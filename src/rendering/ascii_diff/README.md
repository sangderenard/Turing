# ASCII Diff Package

This package began as a minimal copy of the image‑to‑ASCII pipeline from the `timesync` clock demo.
It now also carries the reusable clock and theme utilities so the clock can run directly under
`ascii_diff` without depending on the full `timesync` project.

Available demos:

* **Image diff** – flips an image and renders the changes:

  ```bash
  python ascii_diff/demo.py
  ```

* **Analog clock** – renders an animated clock using theme presets:

  ```bash
  python ascii_diff/clock_demo.py
  ```
