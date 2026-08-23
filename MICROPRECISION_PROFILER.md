# Microprecision profiler

`tools.benchmark_microprecision_matrix` profiles the compiler's materialised
transcendental cores without inserting a width-one/basic-float comparison.
The default permutation is:

- all 19 materialised signal cores;
- limb widths 2, 3, and 4;
- LLVM, C, Fortran, WASM, GLSL, and WebGPU destinations;
- logical batches 1, 8, 64, 512, 4096, and 65536.

## Full run

```powershell
python -u -m tools.benchmark_microprecision_matrix
```

The terminal receives live status and a compact table. The complete records
are written to `build/microprecision-profile/profile.json` and `.csv`.

The backend column is not a fallback selector. Every core is lowered once to
shared SSA and receives universal identities. Only then does the selected
backend own its identity transformations, native emission, deployment
preparation, and launch. Selecting a shader backend does not invoke LLVM. A
backend that cannot preserve an obligation receives an `unsupported` row; a
backend whose route is semantically capable but unfinished receives an
`unavailable` row. Neither receives a borrowed timing.

Each runnable row separates:

- shared lowering and universal-identity time;
- backend emission and native compilation time;
- ABI/deployment preparation time;
- first-launch latency;
- warm minimum, median, and p95 launch time;
- warm nanoseconds per element and per static scalar instruction.

Accuracy uses the exact sum of every returned limb. An exact `Fraction`
evaluation of the same polynomial isolates arithmetic error, while a
high-precision `mpmath` evaluation of the mathematical function includes core
approximation error. Both are reported in local binary64 ULPs, with
correct-rounding rate and median retained bits alongside them.

## Smaller diagnostic run

```powershell
python -u -m tools.benchmark_microprecision_matrix --operators sin exp atan --backends llvm glsl webgpu --widths 3 4 --sizes 1 64 4096 --accuracy-samples 512
```

## Four-limb Mandelbrot

```powershell
python -u -m tools.benchmark_microprecision_matrix --mandelbrot-only --mandelbrot build/microprecision-profile/mandelbrot-4limb.png
```

`--mandlebrot` is accepted as an alias. The default output is one 1280×720
four-limb fractal with no comparison panels. Its numerical resolution is
derived rather than guessed: the canonical Mandelbrot field is 3 units wide,
so binary64's 53-bit pixel threshold is `3 / 2^53`; the default uses
`3 / 2^54`, exactly one bit beyond it. `--fractal-resolution-bits` changes
that figure, while an explicit `--fractal-span` overrides the derived span.

Pass `--fractal-compare` only when a three-panel one-limb/four-limb/difference
diagnostic is explicitly wanted. Both diagnostic panels call the same
recurrence function over the same high-precision camera list; only the
representation boundary differs. Its source digest is persisted in the
receipt.

Coordinates are split from the high-precision decimal camera into four
distinct limbs before iteration; orbit addition and multiplication use the
repository's `Precision` expansion. The neighboring JSON receipt records
coordinate uniqueness/aliasing, pixel disagreements, camera scale, dimensions,
iteration count, elapsed time, and four-limb width.

Fractal parameters can be changed with `--fractal-width`,
`--fractal-height`, `--fractal-iterations`, `--fractal-center-x`,
`--fractal-center-y`, and `--fractal-span`.
